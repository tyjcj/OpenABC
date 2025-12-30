
"""
筛选 synthesisStatistics.pickle，仅打印关心的 29 个电路的 delay/area（映射后指标）。
索引：0=AND 数，1=反相边，2=逻辑深度，3=映射面积，4=映射延迟。
"""

import pickle
from pathlib import Path

# 需要保留的文件名（去掉 _orig.pt 后映射到 stats 的 key）
wanted_files = [
    "ac97_ctrl_orig.pt",
    "aes_orig.pt",
    "aes_secworks_orig.pt",
    "aes_xcrypt_orig.pt",
    "bp_be_orig.pt",
    "des3_area_orig.pt",
    "dft_orig.pt",
    "dynamic_node_orig.pt",
    "ethernet_orig.pt",
    "fir_orig.pt",
    "fpu_orig.pt",
    "i2c_orig.pt",
    "idft_orig.pt",
    "iir_orig.pt",
    "jpeg_orig.pt",
    "mem_ctrl_orig.pt",
    "pci_orig.pt",
    "picosoc_orig.pt",
    "sasc_orig.pt",
    "sha256_orig.pt",
    "simple_spi_orig.pt",
    "spi_orig.pt",
    "ss_pcm_orig.pt",
    "tinyRocket_orig.pt",
    "tv80_orig.pt",
    "usb_phy_orig.pt",
    "vga_lcd_orig.pt",
    "wb_conmax_orig.pt",
    "wb_dma_orig.pt",
]

# 特殊映射（去掉后缀的名称 -> stats 中的键）
special_map = {
    "tinyRocket": "tiny_rocket",
}

def to_key(filename: str) -> str:
    base = filename.replace("_orig.pt", "")
    return special_map.get(base, base)

pkl = Path("test_pt/synthesisStatistics.pickle")
with pkl.open("rb") as f:
    stats = pickle.load(f, encoding="latin1")

want_keys = [to_key(f) for f in wanted_files]
available = set(stats.keys())
missing = [k for k in want_keys if k not in available]
if missing:
    print("缺失的 key（不在 pickle 中）:", missing)

print("筛选后的电路数量:", len([k for k in want_keys if k in available]))

for key in want_keys:
    if key not in stats:
        continue
    arrs = stats[key]
    area = arrs[3][:5]
    delay = arrs[4][:5]
    print(f"\n[{key}] recipes: {len(arrs[0])}")
    print("  area(映射后) 前5:", area)
    print("  delay(映射后) 前5:", delay)

