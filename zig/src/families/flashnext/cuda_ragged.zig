//! Opt-in shared verify capacity. Keep every admitted window within one target pass.
const std = @import("std");

pub const Capacity = struct {
    shared: bool = false,
    rows: usize,
    streams: u32,
};

pub fn wanted() bool {
    const v = std.c.getenv("TF_FLASHNEXT_RAGGED_VERIFY") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Reserve worst-case windows before admission; small chunks and two ranks retain the existing path.
pub fn capacity(on: bool, ranked: bool, chunk: usize, window: usize, legacy: usize, requested: usize) Capacity {
    if (!on or ranked or chunk < 2 * window) return .{ .rows = legacy, .streams = 32 };
    const rows = @min(chunk, @min(32, @max(1, requested)) * window);
    return .{ .shared = true, .rows = rows, .streams = @intCast(rows / window) };
}

test "ragged verify capacity: every admitted window fits one pass" {
    for ([_]usize{ 64, 65, 159, 160, 256, 1024, 1536, 16384 }) |chunk| {
        const c = capacity(true, false, chunk, 32, 64, 32);
        try std.testing.expect(c.rows <= chunk);
        try std.testing.expect(c.streams * 32 <= c.rows);
        try std.testing.expect(c.streams >= 2 and c.streams <= 32);
    }
    const five = capacity(true, false, 1536, 32, 64, 5);
    try std.testing.expectEqual(@as(usize, 160), five.rows);
    try std.testing.expectEqual(@as(usize, 32), capacity(true, false, 1536, 32, 64, 1).rows);
    try std.testing.expectEqual(@as(u32, 5), five.streams);
    for ([_]Capacity{ capacity(false, false, 1536, 32, 64, 32), capacity(true, true, 1536, 32, 64, 32), capacity(true, false, 32, 32, 64, 32) }) |c| {
        try std.testing.expectEqual(@as(usize, 64), c.rows);
        try std.testing.expectEqual(@as(u32, 32), c.streams);
    }
}
