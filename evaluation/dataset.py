"""Evaluation dataset for the DibantuAI agent (Step 6).

Every case pairs a realistic business request with a fully scripted
GLM trajectory: the tool calls GLM is assumed to request (one turn may
request several) and its final reply. The evaluator replays each
trajectory through the real Agent loop and the real registry tools
against the freshly seeded mock store, so runs are deterministic,
offline, and free.

Conventions that keep every case honest:

- ``expected_tools``/``expected_tool_count`` describe the scripted
  trajectory, which the agent must actually execute.
- ``grounding_facts`` are facts that must appear in the final reply
  AND in the tool results the agent really received — scripted replies
  must never state numbers the tools did not return.
- ``expected_state`` is checked through the public business tools after
  the run (product stocks and the 30-day order count; the seed has 4
  orders, so ``monthly_orders=5`` asserts an order was created).

Seed reference (restored before every case): Kopi Susu 24 @18000,
Americano 40 @22000, Croissant 8 @25000, Matcha Latte 15 @24000,
Teh Manis 30 @10000; 4 seeded orders (daily: 1 order / 36000, weekly:
2 orders / 60000, monthly: 4 orders / 176000); customers Budi Santoso,
Siti Rahma, Dewi Lestari, Andi Wijaya.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CASES",
    "CATEGORIES",
    "EvalCase",
    "GLMTurn",
    "StateExpectation",
    "ToolCall",
    "load_cases",
]

#: Every category used by the dataset (spec: Step 6).
CATEGORIES: tuple[str, ...] = (
    "stock_check",
    "order",
    "multi_product_order",
    "insufficient_stock",
    "customer_lookup",
    "sales_report",
    "low_stock",
    "clarification",
    "invalid_request",
    "tool_error",
    "multi_step",
)


@dataclass(frozen=True)
class ToolCall:
    """One tool call the scripted GLM turn requests."""

    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class GLMTurn:
    """One scripted GLM response: tool calls and/or final text.

    A turn with ``tool_calls`` is answered with tool results; a turn
    with only ``text`` ends the trajectory.
    """

    tool_calls: tuple[ToolCall, ...] = ()
    text: str | None = None


@dataclass(frozen=True)
class StateExpectation:
    """Expected mock-store state after a case, read via public tools."""

    #: canonical product name -> expected stock after the run.
    stock: dict[str, int] = field(default_factory=dict)
    #: expected order count in the 30-day window (seed: 4).
    monthly_orders: int = 4


@dataclass(frozen=True)
class EvalCase:
    """One evaluation case: request + scripted trajectory + expectations."""

    id: str
    category: str
    user_message: str
    expected_behavior: str
    expected_tools: tuple[str, ...]
    expected_tool_count: tuple[int, int]  # inclusive (min, max)
    glm_turns: tuple[GLMTurn, ...]
    grounding_facts: tuple[str, ...] = ()
    expected_state: StateExpectation = field(default_factory=StateExpectation)
    expects_tool_error: bool = False


# ---------------------------------------------------------------------------
# Small builders to keep the case table readable
# ---------------------------------------------------------------------------


def tools(*calls: tuple[str, dict[str, Any]]) -> GLMTurn:
    """Build a GLM turn requesting one or more tools."""
    return GLMTurn(tool_calls=tuple(ToolCall(name, input_) for name, input_ in calls))


def say(text: str) -> GLMTurn:
    """Build a final text-only GLM turn."""
    return GLMTurn(text=text)


def case(
    cid: str,
    category: str,
    message: str,
    behavior: str,
    expected_tools: tuple[str, ...],
    tool_count: tuple[int, int],
    turns: tuple[GLMTurn, ...],
    facts: tuple[str, ...] = (),
    stock: dict[str, int] | None = None,
    monthly_orders: int = 4,
    tool_error: bool = False,
) -> EvalCase:
    """Build an EvalCase with a StateExpectation."""
    return EvalCase(
        id=cid,
        category=category,
        user_message=message,
        expected_behavior=behavior,
        expected_tools=expected_tools,
        expected_tool_count=tool_count,
        glm_turns=turns,
        grounding_facts=facts,
        expected_state=StateExpectation(
            stock=stock or {}, monthly_orders=monthly_orders
        ),
        expects_tool_error=tool_error,
    )


CASES: list[EvalCase] = [
    # --- stock_check -------------------------------------------------------
    case(
        "ev_001",
        "stock_check",
        "Apakah stok kopi susu cukup untuk 20 pesanan?",
        "Cek stok Kopi Susu dan jawab dari data aktual.",
        ("check_stock",),
        (1, 1),
        (
            tools(("check_stock", {"product_name": "Kopi Susu"})),
            say("Stok Kopi Susu 24 unit — cukup untuk 20 pesanan."),
        ),
        facts=("24", "Kopi Susu"),
        stock={"Kopi Susu": 24},
    ),
    case(
        "ev_002",
        "stock_check",
        "Berapa sisa croissant?",
        "Laporkan sisa stok Croissant dari data aktual.",
        ("check_stock",),
        (1, 1),
        (
            tools(("check_stock", {"product_name": "Croissant"})),
            say("Sisa Croissant 8 unit."),
        ),
        facts=("8",),
        stock={"Croissant": 8},
    ),
    case(
        "ev_003",
        "stock_check",
        "Stok americano berapa sekarang?",
        "Laporkan stok Americano dari data aktual.",
        ("check_stock",),
        (1, 1),
        (
            tools(("check_stock", {"product_name": "Americano"})),
            say("Stok Americano 40 unit."),
        ),
        facts=("40",),
        stock={"Americano": 40},
    ),
    case(
        "ev_004",
        "stock_check",
        "Cek stok teh manis dong.",
        "Laporkan stok Teh Manis dari data aktual.",
        ("check_stock",),
        (1, 1),
        (
            tools(("check_stock", {"product_name": "Teh Manis"})),
            say("Stok Teh Manis 30 unit."),
        ),
        facts=("30",),
        stock={"Teh Manis": 30},
    ),
    # --- order -------------------------------------------------------------
    case(
        "ev_005",
        "order",
        "Budi pesan 3 kopi susu.",
        "Buat satu pesanan untuk customer Budi dan laporkan hasilnya.",
        ("create_order",),
        (1, 1),
        (
            tools(
                (
                    "create_order",
                    {
                        "customer_name": "Budi Santoso",
                        "items": [{"product_name": "Kopi Susu", "quantity": 3}],
                    },
                )
            ),
            say("Pesanan ORD-0005 dibuat untuk Budi Santoso: 3 Kopi Susu, total 54000."),
        ),
        facts=("ORD-0005", "Kopi Susu"),
        stock={"Kopi Susu": 21},
        monthly_orders=5,
    ),
    case(
        "ev_006",
        "order",
        "Siti Rahma pesan 2 matcha latte.",
        "Buat satu pesanan untuk Siti Rahma dan laporkan hasilnya.",
        ("create_order",),
        (1, 1),
        (
            tools(
                (
                    "create_order",
                    {
                        "customer_name": "Siti Rahma",
                        "items": [{"product_name": "Matcha Latte", "quantity": 2}],
                    },
                )
            ),
            say("Pesanan ORD-0005 dibuat: 2 Matcha Latte untuk Siti Rahma, total 48000."),
        ),
        facts=("48000", "ORD-0005"),
        stock={"Matcha Latte": 13},
        monthly_orders=5,
    ),
    case(
        "ev_007",
        "order",
        "Aldi pesan 1 teh manis.",
        "Daftarkan customer baru Aldi (walk-in) dan buat pesanannya.",
        ("create_order",),
        (1, 1),
        (
            tools(
                (
                    "create_order",
                    {
                        "customer_name": "Aldi",
                        "items": [{"product_name": "Teh Manis", "quantity": 1}],
                    },
                )
            ),
            say("Pesanan ORD-0005 untuk customer baru Aldi: 1 Teh Manis, total 10000."),
        ),
        facts=("Aldi", "10000"),
        stock={"Teh Manis": 29},
        monthly_orders=5,
    ),
    case(
        "ev_008",
        "order",
        "Budi pesan 2 kopi susu, cek stoknya dulu.",
        "Cek stok dulu, lalu buat pesanan jika cukup.",
        ("check_stock", "create_order"),
        (2, 2),
        (
            tools(("check_stock", {"product_name": "Kopi Susu"})),
            tools(
                (
                    "create_order",
                    {
                        "customer_name": "Budi Santoso",
                        "items": [{"product_name": "Kopi Susu", "quantity": 2}],
                    },
                )
            ),
            say("Stok cukup (24). Pesanan ORD-0005 dibuat: 2 Kopi Susu, total 36000."),
        ),
        facts=("24", "36000"),
        stock={"Kopi Susu": 22},
        monthly_orders=5,
    ),
    # --- multi_product_order -----------------------------------------------
    case(
        "ev_009",
        "multi_product_order",
        "Budi pesan 3 kopi susu, 2 croissant, dan 1 americano. Cek stoknya dulu.",
        "Cek stok setiap produk secara berurutan, lalu buat SATU pesanan berisi semua item.",
        ("check_stock", "create_order"),
        (4, 4),
        (
            tools(("check_stock", {"product_name": "Kopi Susu"})),
            tools(("check_stock", {"product_name": "Croissant"})),
            tools(("check_stock", {"product_name": "Americano"})),
            tools(
                (
                    "create_order",
                    {
                        "customer_name": "Budi Santoso",
                        "items": [
                            {"product_name": "Kopi Susu", "quantity": 3},
                            {"product_name": "Croissant", "quantity": 2},
                            {"product_name": "Americano", "quantity": 1},
                        ],
                    },
                )
            ),
            say(
                "Semua stok cukup: Kopi Susu 24, Croissant 8, Americano 40. "
                "ORD-0005 dibuat, total 126000."
            ),
        ),
        facts=("24", "8", "40", "126000"),
        stock={"Kopi Susu": 21, "Croissant": 6, "Americano": 39},
        monthly_orders=5,
    ),
    case(
        "ev_010",
        "multi_product_order",
        "Budi pesan 3 kopi susu, 2 croissant, dan 1 americano.",
        "GLM membatch tiga check_stock dalam satu giliran; semuanya dijalankan.",
        ("check_stock", "create_order"),
        (4, 4),
        (
            tools(
                ("check_stock", {"product_name": "Kopi Susu"}),
                ("check_stock", {"product_name": "Croissant"}),
                ("check_stock", {"product_name": "Americano"}),
            ),
            tools(
                (
                    "create_order",
                    {
                        "customer_name": "Budi Santoso",
                        "items": [
                            {"product_name": "Kopi Susu", "quantity": 3},
                            {"product_name": "Croissant", "quantity": 2},
                            {"product_name": "Americano", "quantity": 1},
                        ],
                    },
                )
            ),
            say("Stok cukup semua (24/8/40). ORD-0005 dibuat, total 126000."),
        ),
        facts=("126000",),
        stock={"Kopi Susu": 21, "Croissant": 6, "Americano": 39},
        monthly_orders=5,
    ),
    case(
        "ev_011",
        "multi_product_order",
        "Siti pesan 1 matcha latte dan 2 teh manis.",
        "Satu pesanan berisi dua item berbeda.",
        ("create_order",),
        (1, 1),
        (
            tools(
                (
                    "create_order",
                    {
                        "customer_name": "Siti Rahma",
                        "items": [
                            {"product_name": "Matcha Latte", "quantity": 1},
                            {"product_name": "Teh Manis", "quantity": 2},
                        ],
                    },
                )
            ),
            say("ORD-0005 dibuat: 1 Matcha Latte dan 2 Teh Manis, total 44000."),
        ),
        facts=("44000",),
        stock={"Matcha Latte": 14, "Teh Manis": 28},
        monthly_orders=5,
    ),
    # --- insufficient_stock --------------------------------------------------
    case(
        "ev_012",
        "insufficient_stock",
        "Budi pesan 100 kopi susu.",
        "Cek stok, temukan kekurangan, JANGAN buat pesanan, jelaskan kekurangannya.",
        ("check_stock",),
        (1, 1),
        (
            tools(("check_stock", {"product_name": "Kopi Susu"})),
            say("Stok Kopi Susu hanya 24 — tidak cukup untuk 100. Mau kurangi jadi 24?"),
        ),
        facts=("24",),
        stock={"Kopi Susu": 24},
    ),
    case(
        "ev_013",
        "insufficient_stock",
        "Budi pesan 100 kopi susu, langsung buatkan.",
        "create_order ditolak tool; laporkan kegagalan dengan jujur.",
        ("create_order",),
        (1, 1),
        (
            tools(
                (
                    "create_order",
                    {
                        "customer_name": "Budi Santoso",
                        "items": [{"product_name": "Kopi Susu", "quantity": 100}],
                    },
                )
            ),
            say("Pesanan gagal: stok Kopi Susu tersedia 24, tidak cukup untuk 100."),
        ),
        facts=("24",),
        stock={"Kopi Susu": 24},
    ),
    case(
        "ev_014",
        "insufficient_stock",
        "Budi pesan 2 croissant dan 50 matcha latte.",
        "Pesanan multi-item ditolak; tidak ada stok yang berubah (atomik).",
        ("create_order",),
        (1, 1),
        (
            tools(
                (
                    "create_order",
                    {
                        "customer_name": "Budi Santoso",
                        "items": [
                            {"product_name": "Croissant", "quantity": 2},
                            {"product_name": "Matcha Latte", "quantity": 50},
                        ],
                    },
                )
            ),
            say(
                "Pesanan gagal: stok Matcha Latte tersedia 15, tidak cukup "
                "untuk 50. Stok tidak berubah."
            ),
        ),
        facts=("15",),
        stock={"Croissant": 8, "Matcha Latte": 15},
    ),
    # --- customer_lookup -----------------------------------------------------
    case(
        "ev_015",
        "customer_lookup",
        "Cari customer bernama Budi.",
        "Cari customer dengan search_customer dan laporkan datanya.",
        ("search_customer",),
        (1, 1),
        (
            tools(("search_customer", {"name": "Budi"})),
            say("Ditemukan 1 customer: Budi Santoso (0812-1111-2222)."),
        ),
        facts=("Budi Santoso", "0812-1111-2222"),
    ),
    case(
        "ev_016",
        "customer_lookup",
        "Customer bernama Adi terdaftar?",
        "Cari 'Adi' dan laporkan hasil kosong dengan jujur.",
        ("search_customer",),
        (1, 1),
        (
            tools(("search_customer", {"name": "Adi"})),
            say("Tidak ada customer dengan nama Adi."),
        ),
        facts=("Adi",),
    ),
    case(
        "ev_017",
        "customer_lookup",
        "Siapa customer bernama Dewi?",
        "Cari 'Dewi' dan laporkan recordnya.",
        ("search_customer",),
        (1, 1),
        (
            tools(("search_customer", {"name": "Dewi"})),
            say("Ditemukan: Dewi Lestari."),
        ),
        facts=("Dewi Lestari",),
    ),
    # --- sales_report --------------------------------------------------------
    case(
        "ev_018",
        "sales_report",
        "Berapa total penjualan hari ini?",
        "Panggil get_sales_report(daily) dan rangkum dari datanya.",
        ("get_sales_report",),
        (1, 1),
        (
            tools(("get_sales_report", {"period": "daily"})),
            say("Hari ini: 1 pesanan, 2 item terjual, pendapatan 36000, terlaris Kopi Susu."),
        ),
        facts=("36000", "Kopi Susu"),
    ),
    case(
        "ev_019",
        "sales_report",
        "Rekap penjualan minggu ini.",
        "Panggil get_sales_report(weekly) dan rangkum dari datanya.",
        ("get_sales_report",),
        (1, 1),
        (
            tools(("get_sales_report", {"period": "weekly"})),
            say("Minggu ini: 2 pesanan, 3 item terjual, pendapatan 60000."),
        ),
        facts=("60000",),
    ),
    case(
        "ev_020",
        "sales_report",
        "Total penjualan bulan ini berapa?",
        "Panggil get_sales_report(monthly) dan rangkum dari datanya.",
        ("get_sales_report",),
        (1, 1),
        (
            tools(("get_sales_report", {"period": "monthly"})),
            say("Bulan ini: 4 pesanan, 11 item terjual, pendapatan 176000."),
        ),
        facts=("176000", "11"),
    ),
    # --- low_stock -----------------------------------------------------------
    case(
        "ev_021",
        "low_stock",
        "Produk apa saja yang stoknya menipis?",
        "Panggil get_low_stock (threshold default 10) dan daftar produknya.",
        ("get_low_stock",),
        (1, 1),
        (
            tools(("get_low_stock", {"threshold": 10})),
            say("Stok menipis: Croissant (8)."),
        ),
        facts=("Croissant", "8"),
    ),
    case(
        "ev_022",
        "low_stock",
        "Produk mana yang stoknya di bawah 16?",
        "Panggil get_low_stock(16) dan daftar produknya.",
        ("get_low_stock",),
        (1, 1),
        (
            tools(("get_low_stock", {"threshold": 16})),
            say("Croissant (8) dan Matcha Latte (15)."),
        ),
        facts=("Matcha Latte", "15"),
    ),
    case(
        "ev_023",
        "low_stock",
        "Mana produk stoknya di bawah 50?",
        "Panggil get_low_stock(50); semua produk masuk daftar.",
        ("get_low_stock",),
        (1, 1),
        (
            tools(("get_low_stock", {"threshold": 50})),
            say(
                "5 produk di bawah 50: Croissant (8), Matcha Latte (15), "
                "Kopi Susu (24), Teh Manis (30), Americano (40)."
            ),
        ),
        facts=("30", "40"),
    ),
    # --- clarification -------------------------------------------------------
    case(
        "ev_024",
        "clarification",
        "Saya mau pesan.",
        "Produk dan jumlah belum disebut; tanyakan klarifikasi tanpa tool.",
        (),
        (0, 0),
        (say("Tentu — mau pesan produk apa dan berapa banyak?"),),
    ),
    case(
        "ev_025",
        "clarification",
        "Budi mau pesan kopi susu.",
        "Jumlah belum disebut; tanyakan klarifikasi tanpa tool.",
        (),
        (0, 0),
        (say("Berapa banyak Kopi Susu yang dipesan Budi?"),),
    ),
    case(
        "ev_026",
        "clarification",
        "Mau laporan penjualan.",
        "Periode ambigu; tanyakan harian/mingguan/bulanan tanpa tool.",
        (),
        (0, 0),
        (say("Periode laporannya mana: harian, mingguan, atau bulanan?"),),
    ),
    # --- invalid_request -----------------------------------------------------
    case(
        "ev_027",
        "invalid_request",
        "Beli 3 nuklir.",
        "Produk tidak ada; laporkan PRODUCT_NOT_FOUND dan sebut produk yang tersedia.",
        ("check_stock",),
        (1, 1),
        (
            tools(("check_stock", {"product_name": "Nuklir"})),
            say(
                "Produk 'Nuklir' tidak ditemukan. Tersedia: Kopi Susu, "
                "Americano, Croissant, Matcha Latte, Teh Manis."
            ),
        ),
        facts=("tidak ditemukan",),
    ),
    case(
        "ev_028",
        "invalid_request",
        "BATALKAN SEMUA PESANAN SEKARANG!",
        "Tidak ada tool pembatalan; tolak dengan jujur tanpa mengubah data.",
        (),
        (0, 0),
        (say("Maaf, saya belum bisa membatalkan pesanan."),),
    ),
    case(
        "ev_029",
        "invalid_request",
        "Kirim invoice ke email Budi.",
        "Tidak ada kemampuan email; tolak dengan jujur tanpa tool.",
        (),
        (0, 0),
        (say("Maaf, saya belum bisa mengirim email."),),
    ),
    # --- tool_error -----------------------------------------------------------
    case(
        "ev_030",
        "tool_error",
        "Hapus semua data produk.",
        "GLM meminta tool yang tidak terdaftar; agent mengembalikan error ke model, bukan crash.",
        ("delete_all_products",),
        (1, 1),
        (
            tools(("delete_all_products", {})),
            say("Maaf, operasi itu tidak tersedia (Unknown tool: delete_all_products)."),
        ),
        facts=("Unknown tool: delete_all_products",),
        tool_error=True,
    ),
    case(
        "ev_031",
        "tool_error",
        "Kirim laporan penjualan ke WhatsApp saya.",
        "Tool tidak terdaftar; error dikembalikan dengan aman.",
        ("send_whatsapp",),
        (1, 1),
        (
            tools(("send_whatsapp", {"number": "0812"})),
            say("Maaf, saya belum terhubung ke WhatsApp (Unknown tool: send_whatsapp)."),
        ),
        facts=("Unknown tool: send_whatsapp",),
        tool_error=True,
    ),
    case(
        "ev_032",
        "tool_error",
        "Cuaca hari ini gimana?",
        "Permintaan di luar domain bisnis; tool tidak dikenal ditangani aman.",
        ("get_weather",),
        (1, 1),
        (
            tools(("get_weather", {"city": "Jakarta"})),
            say("Maaf, saya tidak menangani cuaca (Unknown tool: get_weather)."),
        ),
        facts=("Unknown tool: get_weather",),
        tool_error=True,
    ),
    # --- multi_step -----------------------------------------------------------
    case(
        "ev_033",
        "multi_step",
        "Cari customer Budi, cek stok kopi susu, lalu buatkan pesanan 1 kopi susu untuknya.",
        "Tiga langkah berurutan: cari customer, cek stok, buat pesanan.",
        ("search_customer", "check_stock", "create_order"),
        (3, 3),
        (
            tools(("search_customer", {"name": "Budi"})),
            tools(("check_stock", {"product_name": "Kopi Susu"})),
            tools(
                (
                    "create_order",
                    {
                        "customer_name": "Budi Santoso",
                        "items": [{"product_name": "Kopi Susu", "quantity": 1}],
                    },
                )
            ),
            say(
                "Budi Santoso ditemukan. Stok Kopi Susu 24. "
                "ORD-0005 dibuat: 1 Kopi Susu, total 18000."
            ),
        ),
        facts=("Budi Santoso", "24", "18000"),
        stock={"Kopi Susu": 23},
        monthly_orders=5,
    ),
    case(
        "ev_034",
        "multi_step",
        "Cek stok croissant; kalau kurang dari 10, tambahkan stok 20.",
        "Cek stok lalu restock lewat update_stock hanya bila di bawah ambang.",
        ("check_stock", "update_stock"),
        (2, 2),
        (
            tools(("check_stock", {"product_name": "Croissant"})),
            tools(("update_stock", {"product_name": "Croissant", "quantity_change": 20})),
            say("Stok Croissant 8 (di bawah 10). Ditambah 20 menjadi 28."),
        ),
        facts=("8", "28"),
        stock={"Croissant": 28},
    ),
]


def load_cases() -> list[EvalCase]:
    """Return every evaluation case (fresh list, cases are immutable)."""
    return list(CASES)
