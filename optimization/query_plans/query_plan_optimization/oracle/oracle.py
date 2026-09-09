import os
import time
from query_plan_timing import event
from dataclasses import dataclass

import networkx as nx  # type: ignore
import psycopg  # type: ignore
from codec.codec import AliasesCodec, Codec, JoinTree, JoinTreeBranch, JoinTreeLeaf
from workload.workloads import WorkloadSpec, WorkloadSpecDefinition

from .batch_gate import serialized_batch_query

from .structures import (
    CompletedQuery,
    FailedQuery,
    QueryExecutionSpec,
    QueryResult,
    TimedOutQuery,
)

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


@dataclass
class WorkloadInput:
    id: str
    encoded_query: list[int]
    timeout_secs: float


def _resolve_codec(workload: WorkloadSpec) -> Codec:
    return AliasesCodec(workload.all_tables)


def _default_plan(workload: WorkloadSpecDefinition) -> str:
    return workload.query_template.format(
        ", ".join(
            [
                f"{table} AS {table}{alias_num}"
                for (table, num_aliases) in workload.query_tables
                for alias_num in range(1, num_aliases + 1)
            ]
        )
    )


def _decode_query(workload: WorkloadSpec, encoded: list[int]) -> str:
    codec = _resolve_codec(workload)
    join_tree = codec.decode(workload.query_tables, encoded)
    join_clause = join_tree.to_join_clause()

    if workload.pg_hint_plan_join_order:
        return (
            f"/*+\nLeading({join_tree.to_order_hint()})\n{join_tree.to_operator_hint()}\n*/\n{workload.query_template}"
        )
    return f"/*+\n{join_tree.to_operator_hint()}\n*/\n{workload.query_template.format(join_clause)}"


def plan_has_crossjoin(workload: WorkloadSpec, encoded: list[int]) -> bool:
    codec = _resolve_codec(workload)
    join_tree = codec.decode(workload.query_tables, encoded)
    return _join_tree_has_crossjoin(workload, join_tree)


def _crossjoin_at_branch(workload: WorkloadSpec, left: JoinTree, right: JoinTree) -> bool:
    left_aliases = left.tables_aliases()
    right_aliases = right.tables_aliases()

    filtered_graph = nx.subgraph_view(
        workload.schema.query_join_graph,
        filter_node=lambda n: n in left_aliases or n in right_aliases,
        filter_edge=lambda u, v: (u in left_aliases or u in right_aliases)
        and (v in left_aliases or v in right_aliases),
    )

    if not nx.is_connected(filtered_graph):
        return True
    return False


def _join_tree_has_crossjoin(workload: WorkloadSpec, join_tree: JoinTree) -> bool:
    match join_tree:
        case JoinTreeLeaf(_, _):
            return False
        case JoinTreeBranch(left, right, _):
            return (
                _crossjoin_at_branch(workload, left, right)
                or _join_tree_has_crossjoin(workload, left)
                or _join_tree_has_crossjoin(workload, right)
            )
        case _:
            raise ValueError("Unknown join tree type")


@serialized_batch_query
def _execute_query(spec: WorkloadSpec, input: WorkloadInput) -> QueryResult:
    query = _decode_query(spec, input.encoded_query)
    # print(query)
    timeout_ms = input.timeout_secs * 1000
    execution_spec = QueryExecutionSpec(id=input.id, query=query, timeout_secs=input.timeout_secs)

    with psycopg.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD) as conn:
        with conn.cursor() as cur:
            query_clock = None
            try:
                cur.execute(f"SET statement_timeout TO {timeout_ms}")
                cur.execute("SET client_encoding TO 'UTF8'")
                start_time = time.time()
                query_clock = time.perf_counter()
                cur.execute(query)
                end_time = time.time()
                event("db_query_execution", time.perf_counter() - query_clock, result="completed")

                return CompletedQuery(
                    spec=execution_spec,
                    elapsed_secs=end_time - start_time,
                )
            except psycopg.errors.QueryCanceled:
                if query_clock is not None:
                    event("db_query_execution", time.perf_counter() - query_clock, result="cancelled", timeout_secs=input.timeout_secs)
                return TimedOutQuery(spec=execution_spec, elapsed_secs=timeout_ms / 1000)
            except Exception as e:
                end_time = time.time()
                if query_clock is not None:
                    event("db_query_execution", time.perf_counter() - query_clock, result="failed", error_type=type(e).__name__)
                return FailedQuery(
                    spec=execution_spec,
                    elapsed_secs=end_time - start_time,
                    error=str(e),
                )


def oracle(workload: WorkloadSpec, workload_inputs: list[WorkloadInput]) -> list[QueryResult]:
    results = []
    for input in workload_inputs:
        result = _execute_query(workload, input)
        results.append(result)
    return results


if __name__ == "__main__":
    from workload.workloads import get_workload_set

    workload_set = get_workload_set("JOB")
    for query_name, spec in workload_set.queries.items():
        result = oracle(
            spec,
            [WorkloadInput(id=query_name, encoded_query=[1, 2, 3], timeout_secs=5)],
        )
        match result[0]:
            case CompletedQuery(spec=_, elapsed_secs=elapsed_secs):
                print(f"{query_name}: completed in {elapsed_secs}s")
            case TimedOutQuery(spec=_, elapsed_secs=elapsed_secs):
                print(f"{query_name}: timed out after {elapsed_secs}s")
            case FailedQuery(spec=_, elapsed_secs=elapsed_secs, error=error):
                print(f"{query_name}: failed after {elapsed_secs}s: {error}")
