use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;

use super::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};
use crate::coalesce_batches::CoalesceBatchesExec;
use crate::metrics::MetricsSet;
use crate::projection::ProjectionExec;
use crate::stream::RecordBatchStreamAdapter;
use crate::union::UnionExec;
use crate::CoalescePartitionsExec;

use crate::filter::FilterExec;
use crate::insert::make_count_schema;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use arrow_array::{ArrayRef, UInt64Array};
use datafusion_common::{exec_err, internal_err, Result};
use datafusion_execution::TaskContext;
use datafusion_expr::FilterOp;
use datafusion_physical_expr::{
    Distribution, EquivalenceProperties, PhysicalExpr, PhysicalSortRequirement,
};
// use crate::delete::DeleteSinkExec;

use async_trait::async_trait;
use futures::StreamExt;

/// `OverwriteSink` implements updating streams of [`RecordBatch`]es in
/// user defined destinations.
///
/// The `Display` impl is used to format the sink for explain plan
/// output.
#[async_trait]
pub trait OverwriteSink: DisplayAs + Debug + Send + Sync {
    /// Returns the data sink as [`Any`](std::any::Any) so that it can be
    /// downcast to a specific implementation.
    fn as_any(&self) -> &dyn Any;

    /// Return a snapshot of the [MetricsSet] for this
    /// [OverwriteSink].
    ///
    /// See [ExecutionPlan::metrics()] for more details
    fn metrics(&self) -> Option<MetricsSet>;

    // TODO add desired input ordering
    // How does this sink want its input ordered?

    /// Writes the data to the sink, returns the number of values written
    ///
    /// This method will be called exactly once during each DML
    /// statement. Thus prior to return, the sink should do any commit
    /// or rollback required.
    async fn overwrite_with(
        &self,
        input_data: SendableRecordBatchStream,
        context: &Arc<TaskContext>,
        filter: Option<Arc<dyn PhysicalExpr>>,
        op: FilterOp,
    ) -> Result<u64>;
}

/// Execution plan for updating record batches to a [`OverwriteSink`]
///
/// Returns a single row with the number of values updated
pub struct UpdateSinkExec {
    /// Input plan that produces the record batches to be updated.
    input_plan: Arc<dyn ExecutionPlan>,
    /// Input plan that produces the record batches to be updated.
    // prune_plan: Arc<dyn ExecutionPlan>,
    /// Sink to which to update
    sink: Arc<dyn OverwriteSink>,
    /// Schema of the sink for validating the input data
    sink_schema: SchemaRef,
    /// Schema describing the structure of the output data.
    count_schema: SchemaRef,
    /// Optional required sort order for output data.
    sort_order: Option<Vec<PhysicalSortRequirement>>,
    cache: PlanProperties,
}

impl fmt::Debug for UpdateSinkExec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "UpdateSinkExec schema: {:?}", self.count_schema)
    }
}

impl UpdateSinkExec {
    /// Create a plan to update the `sink`
    pub fn new(
        input_plan: Arc<dyn ExecutionPlan>,
        // prune_plan: Arc<dyn ExecutionPlan>,
        sink: Arc<dyn OverwriteSink>,
        sink_schema: SchemaRef,
        sort_order: Option<Vec<PhysicalSortRequirement>>,
    ) -> Self {
        let count_schema = make_count_schema();
        let cache = Self::create_schema(&input_plan, count_schema);
        Self {
            input_plan,
            // prune_plan,
            sink,
            sink_schema,
            count_schema: make_count_schema(),
            sort_order,
            cache,
        }
    }

    fn create_schema(
        input: &Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
    ) -> PlanProperties {
        let eq_properties = EquivalenceProperties::new(schema);
        PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            input.execution_mode(),
        )
    }

    fn execute_input_plan_stream(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input_plan.execute(partition, context)?;

        debug_assert_eq!(
            self.sink_schema.fields().len(),
            self.input_plan.schema().fields().len()
        );

        // Find input columns that may violate the not null constraint.
        let risky_columns: Vec<_> = self
            .sink_schema
            .fields()
            .iter()
            .zip(self.input_plan.schema().fields().iter())
            .enumerate()
            .filter_map(|(i, (sink_field, input_field))| {
                if !sink_field.is_nullable() && input_field.is_nullable() {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if risky_columns.is_empty() {
            Ok(input_stream)
        } else {
            // Check not null constraint on the input stream
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.sink_schema.clone(),
                input_stream
                    .map(move |batch| check_not_null_constraints(batch?, &risky_columns)),
            )))
        }
    }

    /// Input execution plan
    pub fn input_plan(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input_plan
    }

    /// Returns update sink
    pub fn sink(&self) -> &dyn OverwriteSink {
        self.sink.as_ref()
    }

    /// Optional sort order for output data
    pub fn sort_order(&self) -> &Option<Vec<PhysicalSortRequirement>> {
        &self.sort_order
    }

    /// Returns the metrics of the underlying [OverwriteSink]
    pub fn metrics(&self) -> Option<MetricsSet> {
        self.sink.metrics()
    }
}

impl DisplayAs for UpdateSinkExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "UpdateSinkExec: sink=")?;
                self.sink.fmt_as(t, f)
            }
        }
    }
}

impl ExecutionPlan for UpdateSinkExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> SchemaRef {
        self.count_schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // OverwriteSink is responsible for dynamically partitioning its
        // own input at execution time.
        vec![false]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // OverwriteSink is responsible for dynamically partitioning its
        // own input at execution time, and so requires a single input partition.
        vec![Distribution::SinglePartition; self.children().len()]
    }

    fn required_input_ordering(&self) -> Vec<Option<Vec<PhysicalSortRequirement>>> {
        // The required input ordering is set externally (e.g. by a `ListingTable`).
        // Otherwise, there is no specific requirement (i.e. `sort_expr` is `None`).
        vec![self.sort_order.as_ref().cloned()]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        // Maintains ordering in the sense that the updated records will reflect
        // the ordering of the input. For more context, see:
        //
        // https://github.com/apache/arrow-datafusion/pull/6354#discussion_r1195284178
        vec![true]
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input_plan.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            input_plan: children[0].clone(),
            // prune_plan: children[0].clone(),
            sink: self.sink.clone(),
            sink_schema: self.sink_schema.clone(),
            count_schema: self.count_schema.clone(),
            sort_order: self.sort_order.clone(),
            cache: self.cache.clone(),
        }))
    }

    /// Execute the plan and return a stream of `RecordBatch`es for
    /// the specified partition.
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!("UpdateSinkExec can only be called on partition 0!");
        }

        let input_data = self.execute_input_plan_stream(0, context.clone())?;

        let filter_predicate = if let Some(coalesce_partition_exec) = self
            .input_plan
            .as_any()
            .downcast_ref::<CoalescePartitionsExec>()
        {
            let predicate = match coalesce_partition_exec
                .input()
                .as_any()
                .downcast_ref::<UnionExec>()
            {
                Some(coal_part_exec) => {
                    let project_exec = coal_part_exec.inputs()[0]
                        .as_any()
                        .downcast_ref::<ProjectionExec>()
                        .unwrap();
                    let coal_batch_exec = project_exec
                        .input()
                        .as_any()
                        .downcast_ref::<CoalesceBatchesExec>()
                        .unwrap();
                    let filter_exec = coal_batch_exec
                        .input()
                        .as_any()
                        .downcast_ref::<FilterExec>()
                        .unwrap();
                    Some(filter_exec.predicate().clone())
                }
                None => None,
            };
            predicate
        } else {
            None
        };

        let count_schema = self.count_schema.clone();
        let sink = self.sink.clone();

        let stream = futures::stream::once(async move {
            sink.overwrite_with(input_data, &context, filter_predicate, FilterOp::Update)
                .await
                .map(make_count_batch)
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            count_schema,
            stream,
        )))
    }
}

/// Create an output record batch with a count
///
/// ```text
/// +-------+,
/// | count |,
/// +-------+,
/// | 6     |,
/// +-------+,
/// ```
pub fn make_count_batch(count: u64) -> RecordBatch {
    let array = Arc::new(UInt64Array::from(vec![count])) as ArrayRef;

    RecordBatch::try_from_iter_with_nullable(vec![("count", array, false)]).unwrap()
}

fn check_not_null_constraints(
    batch: RecordBatch,
    column_indices: &Vec<usize>,
) -> Result<RecordBatch> {
    for &index in column_indices {
        if batch.num_columns() <= index {
            return exec_err!(
                "Invalid batch column count {} expected > {}",
                batch.num_columns(),
                index
            );
        }

        if batch.column(index).null_count() > 0 {
            return exec_err!(
                "Invalid batch column at '{}' has null but schema specifies non-nullable",
                index
            );
        }
    }
    Ok(batch)
}
