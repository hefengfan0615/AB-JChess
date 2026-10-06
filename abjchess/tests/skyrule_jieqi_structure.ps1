param(
    [string]$SourcePath = (Join-Path $PSScriptRoot '..\src\position.cpp')
)

$ErrorActionPreference = 'Stop'
$source = Get-Content -LiteralPath $SourcePath -Raw

if ($source -notmatch 'struct\s+SkyruleSnapshot\s*\{') {
    throw 'Missing the before/after SkyRule snapshot model.'
}
if ($source -notmatch 'struct\s+SkyruleDelta\s*\{') {
    throw 'Missing the combined SkyRule delta model.'
}
if ($source -notmatch 'visibleTrueRooters') {
    throw 'Missing visible true-root metadata.'
}
if ($source -notmatch 'targetType\s*!=\s*ADVISOR') {
    throw 'Duffish target mapping is missing the ADVISOR case.'
}
if ($source -notmatch 'skyrule_has_overlong_persistent_check') {
    throw 'Missing internal-node persistent-check adjudication helper.'
}

$chasedStart = $source.IndexOf('Bitboard Position::skyrule_chased(')
$chasedEnd = $source.IndexOf('void Position::set_skyrule_info(', $chasedStart)
if ($chasedStart -lt 0 -or $chasedEnd -lt 0) {
    throw 'Could not isolate Position::skyrule_chased().' 
}
$chased = $source.Substring($chasedStart, $chasedEnd - $chasedStart)
$snapshotCalls = [regex]::Matches($chased, 'skyrule_snapshot\s*\(').Count
if ($snapshotCalls -ne 2) {
    throw "Position::skyrule_chased() must compare exactly two snapshots; found $snapshotCalls."
}
if ($chased -notmatch 'skyrule_delta\s*\(') {
    throw 'Position::skyrule_chased() does not reuse the combined before/after delta.'
}

$infoStart = $source.IndexOf('void Position::set_skyrule_info(')
$infoEnd = $source.IndexOf('int Position::skyrule_continuous_chase_count_in_place', $infoStart)
if ($infoStart -lt 0 -or $infoEnd -lt 0) {
    throw 'Could not isolate Position::set_skyrule_info().' 
}
$info = $source.Substring($infoStart, $infoEnd - $infoStart)
if ($info -notmatch 'previousSameSide->skyruleChased') {
    throw 'SkyRule continuity does not use the previous same-side chased set.'
}
if ($info -notmatch 'si->skyruleChased\s*&\s*previousSameSide->skyruleChased') {
    throw 'SkyRule continuity is missing the common-target fast path.'
}

$forbiddenStart = $source.IndexOf('bool Position::skyrule_move_forbidden(')
$forbiddenEnd = $source.IndexOf('bool Position::forbidden_by_skyrule_jieqi', $forbiddenStart)
if ($forbiddenStart -lt 0 -or $forbiddenEnd -lt 0) {
    throw 'Could not isolate Position::skyrule_move_forbidden().' 
}
$forbidden = $source.Substring($forbiddenStart, $forbiddenEnd - $forbiddenStart)
if ($forbidden -notmatch 'if\s*\(move_dark\(m\)\s*\|\|\s*capture\(m\)\)\s*return\s+false') {
    throw 'Dark moves and captures do not bypass the SkyRule forbidden-move path.'
}
if ($forbidden -notmatch 'previousSameSide->skyruleAction\s*==\s*SKYRULE_CHECK[\s\S]*?givesCheck[\s\S]*?return\s+false') {
    throw 'Checking-move fast rejection is missing before temporary do_move().' 
}

$detectStart = $source.IndexOf('Value Position::detect_chases(')
$detectEnd = $source.IndexOf('int Position::skyrule_repeat_count', $detectStart)
if ($detectStart -lt 0 -or $detectEnd -lt 0) {
    throw 'Could not isolate Position::detect_chases().' 
}
$detect = $source.Substring($detectStart, $detectEnd - $detectStart)
if ($detect -notmatch 'skyrule_delta\s*\(') {
    throw 'Position::detect_chases() does not reuse a combined chase delta.'
}
if ($detect -notmatch 'capturedPiece[\s\S]*movedDark[\s\S]*break') {
    throw 'Cycle scan does not stop at capture/dark identity boundaries.'
}
if ($detect -notmatch 'persistentTargets[\s\S]*snapshotTargets') {
    throw 'Persistent chase tracking does not intersect complete snapshots.'
}
if ($detect -notmatch 'indirect_counter_chase_cycle') {
    throw 'Indirect counter-chase cycle adjudication is missing.'
}
if ($detect -notmatch 'anchor_target_counter_chase_cycle') {
    throw 'Anchor target counter-chase adjudication is missing.'
}
if ($detect -notmatch 'reciprocal_actual_chase_retreat') {
    throw 'Stable reciprocal chase-retreat adjudication is missing.'
}
if ($detect -notmatch 'independent_chase_retreat_reply') {
    throw 'Independent chase-retreat priority handling is missing.'
}
if ($detect -notmatch 'persistent_mixed_chasing') {
    throw 'Narrow persistent mixed-check/chase classification is missing.'
}
if ($detect -notmatch 'mixedCheckAgainstReturningChasedPiece') {
    throw 'Returning-check versus persistent chase adjudication is missing.'
}
if ($detect -notmatch 'checks_between_persistent_chase_escapes') {
    throw 'Checks-between-persistent-chase-escapes adjudication is missing.'
}
if ($detect -notmatch '!\(checkUs\s*\|\|\s*chaseUs\)\s*&&\s*activePersistentThem') {
    throw 'Opponent persistent-chase root delay is missing.'
}
if ($detect -notmatch 'paired_split_chase_against_bounding_check') {
    throw 'Paired split-chase versus bounding-check adjudication is missing.'
}
if ($detect -notmatch 'indirect_king_reversal_chase') {
    throw 'Indirect king-reversal chase exception is missing.'
}
if ($source -notmatch 'ply\s*>\s*0[\s\S]*mate_in\(ply\s*-\s*1\)') {
    throw 'Root/internal SkyRule mate timing guard is missing.'
}
if ($source -notmatch 'if\s*\(st->movedDark\)\s*\r?\n\s*byTypeBB\[DARK\]') {
    throw 'Undo visibility restoration is not keyed to movedDark metadata.'
}
if ($info -notmatch 'si->previous->capturedPiece[\s\S]*si->previous->movedDark') {
    throw 'Same-side SkyRule continuity can cross a capture/dark boundary.'
}
if ($forbidden -notmatch '!st->move\.is_ok\(\)[\s\S]*st->capturedPiece[\s\S]*st->movedDark') {
    throw 'Forbidden-move filtering can cross an intervening capture/dark boundary.'
}

Write-Host 'PASS: Duffish SkyRule structure assertions.'
