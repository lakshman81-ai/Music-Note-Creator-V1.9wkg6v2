from typing import List
from .models import NoteEvent, AnalysisData

def apply_theory(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
) -> List[NoteEvent]:
    """
    Stage C: Minimal, non-destructive theory application.
    """
    # Requirements:
    # - Attach notes to analysis_data if desired (we did this in stage_a/b or orchestrator,
    #   but we can ensure consistency here).
    # - MUST NOT alter pitches, start times, or durations.
    # - Return events unchanged.

    # Update analysis_data.notes to match events, just in case they were modified elsewhere or need syncing.
    # The orchestrator sets this up, but it's safe to re-assign or ensure they are the same objects.
    analysis_data.notes = events

    # Future expansion: key analysis, harmonic analysis, etc. could go here.

    return events
