# Third-party license notes — `third_party/openjtalk/`

Vendored source of **Open JTalk 1.11** (C text-to-speech frontend), compiled
directly by the root `CMakeLists.txt` into the static lib `openjtalk`.
Each component keeps its original copyright/license notice in its source
files; this file is only a navigation aid.

| Component | Path | Copyright | License |
|---|---|---|---|
| OpenJTalk C frontend (`jpcommon/`, `njd/`, `njd2jpcommon/`, `mecab2njd/`, `text2mecab/`, `njd_set_*`) | per-file headers | © 2008-2016 Nagoya Institute of Technology | BSD-style (see headers) |
| MeCab C++ runtime | `mecab/src/` | © 2001-2011 Taku Kudo, © 2004-2006 NTT | tri-licensed: BSD / GPL / LGPL (see headers) |
| liblbfgs (embedded in `mecab/src/lbfgs.*`) | `mecab/src/lbfgs.h` | © Naoaki Okazaki | MIT |

Upstream: Open JTalk — https://open-jtalk.sourceforge.net/
(MeCab — https://taku910.github.io/mecab/ ; liblbfgs — https://github.com/chokkan/liblbfgs)

Note: the `*_rule_euc_jp.h` / `*_rule_shift_jis.h` charset-rule data tables
carry no per-file notice in the upstream source archive either; only the
`*_rule_utf_8.h` variants are actually compiled (`CHARSET_UTF_8`).
