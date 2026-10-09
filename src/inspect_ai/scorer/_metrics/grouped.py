from collections.abc import Mapping
from typing import Literal, cast

from inspect_ai._util.registry import is_registry_object, registry_unqualified_name
from inspect_ai.scorer._metric import (
    Metric,
    MetricProtocol,
    SampleScore,
    Value,
    ValueToFloat,
    metric,
    metric_scores,
    value_to_float,
)


@metric
def grouped(
    metric: Metric,
    group_key: str,
    *,
    all: Literal["samples", "groups"] | Literal[False] = "samples",
    all_label: str = "all",
    value_to_float: ValueToFloat = value_to_float(),
    name_template: str = "{group_name}",
) -> Metric:
    """
    Creates a grouped metric that applies the given metric to subgroups of samples.

    Args:
      metric: The metric to apply to each group of samples.
      group_key: The metadata key used to group samples. Each sample must have this key in its metadata.
      all: How to compute the "all" aggregate score:
          - "samples": Apply the metric to all samples regardless of groups
          - "groups": Calculate the mean of all group scores (requires a metric
            that returns a single value per group)
          - False: Don't calculate an aggregate score
      all_label: The label for the "all" key in the returned dictionary.
      value_to_float: Function to convert metric values to floats, used when all="groups".
      name_template: Template for the name of each group. The default is "{group_name}".

    Returns:
        A new metric function that returns a dictionary mapping group names to their scores,
        with an optional "all" key for the aggregate score. If the metric returns a
        dictionary (e.g. `ci()`), each of its entries is reported under
        "{group name}-{key}" (e.g. "math-lower" and "math-upper").

    Raises:
        ValueError: If the metric is declared with `scores="unreduced"` (e.g.
          `frequency()`), which `grouped()` does not pass through.
    """
    metric_label = (
        f"{registry_unqualified_name(metric)}()"
        if is_registry_object(metric)
        else "the metric"
    )
    if is_registry_object(metric) and metric_scores(metric) == "unreduced":
        raise ValueError(
            f"grouped() does not support {metric_label}, which is declared with "
            'scores="unreduced": grouped() would receive epoch-reduced scores.'
        )

    def grouped_metric(scores: list[SampleScore]) -> Value:
        import numpy as np

        # Satisfy the type checker that the metric is a MetricProtocol
        metric_protocol = cast(MetricProtocol, metric)

        # Own the empty case: an all-unscored task must still report the
        # metric's shape (the aggregate key) rather than degrading to a bare
        # NaN synthesized upstream (#5150, the #4718 precedent).
        if not scores:
            if not all:
                return cast(Value, {})
            return cast(Value, {all_label: float("nan")})

        # Slice the scores into groups
        scores_dict: dict[str, list[SampleScore]] = {}
        for sample_score in scores:
            if (
                sample_score.sample_metadata is None
                or group_key not in sample_score.sample_metadata
            ):
                raise ValueError(
                    f"Sample {sample_score.sample_id} has no {group_key} metadata. To compute a grouped metric each sample metadata must have a value for '{group_key}'"
                )
            group_name = str(sample_score.sample_metadata.get(group_key))
            if group_name not in scores_dict:
                scores_dict[group_name] = []
            scores_dict[group_name].append(sample_score)

        # Compute the per group metric
        grouped_scores: dict[str, Value] = {}
        for group_name, values in scores_dict.items():
            try:
                group_score = metric_protocol(values)
            except ValueError as ex:
                # the metric's own message cannot say which group it failed on
                raise ValueError(
                    f"{metric_label} failed for the samples with "
                    f"{group_key}={group_name!r}: {ex}"
                ) from ex
            grouped_scores[name_template.format(group_name=group_name)] = group_score

        if not all:
            return _flatten(grouped_scores)
        else:
            # Guard against a group whose name collides with all_label —
            # otherwise the per-group metric would be silently overwritten
            # by the aggregate below.
            if all_label in grouped_scores:
                raise ValueError(
                    f"The group name '{all_label}' collides with the `all_label` "
                    f"used for the aggregate score. Pass a different `all_label` "
                    f"to grouped() (or set all=False) to avoid overwriting this "
                    f"group's metric."
                )
            # Compute the all metric
            all_group_metric = None
            if all == "samples":
                # samples means apply the metric to all samples
                all_group_metric = metric_protocol(scores)
            elif all == "groups":
                # group means the overall score is the mean of all the group scores
                if not grouped_scores:
                    all_group_metric = 0.0
                elif any(isinstance(val, Mapping) for val in grouped_scores.values()):
                    raise ValueError(
                        f'grouped(all="groups") cannot average {metric_label}, which '
                        'returns a dictionary for each group. Use all="samples" or '
                        "all=False with this metric."
                    )
                else:
                    all_group_metric = np.mean(
                        [value_to_float(val) for val in grouped_scores.values()]
                    ).item()

            return _flatten({**grouped_scores, all_label: all_group_metric})

    return grouped_metric


def _flatten(grouped_scores: Mapping[str, Value | None]) -> Value:
    """Report each entry of a dictionary-valued group score as `{name}-{key}`.

    A metric value can be a dictionary of scalars but not a dictionary of
    dictionaries, so a metric such as `ci()` cannot be nested under its group.
    """
    flat: dict[str, Value | None] = {}
    for name, value in grouped_scores.items():
        entries = (
            {f"{name}-{key}": item for key, item in value.items()}
            if isinstance(value, Mapping)
            else {name: value}
        )
        for key, item in entries.items():
            if key in flat:
                raise ValueError(
                    f"grouped() produced the metric name '{key}' twice. Pass a "
                    "`name_template` or `all_label` that keeps the names distinct."
                )
            flat[key] = item
    return cast(Value, flat)
