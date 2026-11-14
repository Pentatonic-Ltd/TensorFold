//! Explicit MTP controls share one cap and confidence convention across drivers and common lane rounds.
const std = @import("std");
pub const Controls = struct {
    cap: ?u32 = null,
    floor: ?f64 = null,
    pub fn limit(p: Controls, maximum: usize) usize {
        return if (p.cap) |n| @min(n, maximum) else maximum;
    }
    pub fn configure(p: Controls, rule: anytype) void {
        rule.most = p.limit(rule.most);
        rule.floor = p.floor;
    }
    /// A chain always verifies its first offered draft; subsequent cumulative chances must reach the floor.
    pub fn prefix(p: Controls, chances: []const f64) usize {
        const n = p.limit(chances.len);
        const floor = p.floor orelse return n;
        if (n == 0) return 0;
        var kept: usize = 1;
        for (chances[1..n]) |chance| {
            if (!(chance >= floor)) break;
            kept += 1;
        }
        return kept;
    }
};

/// Acceptance gate: a stream whose cumulative draft acceptance has fallen below `gate_floor`
/// (after `gate_warmup` drafted rows) decodes one token a round instead of drafting.
pub const gate_floor: f64 = 0.5;
pub const gate_warmup: u64 = 16;
/// While gated, one round in `gate_probe` still drafts a single row so a recovering stream can warm up again.
pub const gate_probe: u64 = 8;

pub fn cold(drafted: u64, accepted: u64) bool {
    if (drafted < gate_warmup) return false;
    return @as(f64, @floatFromInt(accepted)) < gate_floor * @as(f64, @floatFromInt(drafted));
}

/// Drafts to hold for the next round: the asked depth, none when cold, one on a probe round.
pub fn gatedDepth(depth: u32, drafted: u64, accepted: u64, rounds: u64) u32 {
    if (depth == 0 or !cold(drafted, accepted)) return depth;
    return if (rounds % gate_probe == 0) 1 else 0;
}

/// The backend reports per-level probabilities; the common allocator consumes cumulative chain chances.
pub fn cumulative(raw: []const f32, out: []f64) !void {
    if (raw.len != out.len) return error.BadDraftCount;
    var chance: f64 = 1;
    for (raw, out) |p, *v| {
        if (!(p >= 0 and p <= 1)) return error.BadDraftProbability;
        chance *= p;
        v.* = chance;
    }
}

test "mtp policy: cap and cumulative confidence retain the first draft and both endpoints" {
    const chances = [_]f64{ 0.8, 0.64, 0.512, 0.4096 };
    for ([_]f64{ 0, 0.5, 1 }, [_]usize{ 4, 3, 1 }) |floor, want| {
        try std.testing.expectEqual(want, (Controls{ .floor = floor }).prefix(&chances));
    }
    try std.testing.expectEqual(@as(usize, 2), (Controls{ .floor = 0, .cap = 2 }).prefix(&chances));
    try std.testing.expectEqual(@as(usize, 0), (Controls{ .cap = 0 }).prefix(&chances));
    var out: [3]f64 = undefined;
    try cumulative(&.{ 0.5, 0.5, 0.5 }, &out);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 0.25, 0.125 }, &out);
    try std.testing.expectError(error.BadDraftProbability, cumulative(&.{std.math.nan(f32)}, out[0..1]));
}

test "mtp policy: the acceptance gate drafts when warm, steps when cold, probes while cold" {
    try std.testing.expect(!cold(0, 0));
    try std.testing.expect(!cold(15, 0)); // too few drafts to judge
    try std.testing.expect(!cold(20, 10)); // exactly at the floor stays warm
    try std.testing.expect(cold(20, 9));
    try std.testing.expectEqual(@as(u32, 4), gatedDepth(4, 100, 90, 3));
    try std.testing.expectEqual(@as(u32, 0), gatedDepth(4, 100, 20, 3));
    try std.testing.expectEqual(@as(u32, 1), gatedDepth(4, 100, 20, 8));
    try std.testing.expectEqual(@as(u32, 0), gatedDepth(0, 100, 20, 8));
}
