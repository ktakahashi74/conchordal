# 部分音群モデルv2の検証記録

2026-09-29。[取得前契約](body-fitness-partial-groups-contract-20260929.md)による隔離試作。[主計画](body-aware-fitness-plan.md)のF2内で、実子の部分音から身体評価を作る方式を調べた。本番runtimeへは接続していない。

## 結論

**v2は内部式の検査に通ったが、旧参照との精度条件と対象負荷の費用条件に不合格。** 小さい身体では局所予算に収まったが、全family・対象負荷の同hop出生は成立していない。合格した条件だけを本番へ接続しない。

| 判定 | 結果 | 事前条件 |
|---|---:|---:|
| 新式・入力境界・実身体種別のunit | debug/releaseとも10成功 | 全件成功 |
| 旧728候補中の誤差上限超過 | 159件 | 0件 |
| score最大絶対差 | 0.3971686363 | 0.025以下 |
| level最大絶対差 | 0.1268229783 | 0.0125以下 |
| 旧score差0.1以上の順位逆転 | 3対 | 0対 |
| 費用条件 | 10/20成功 | 20/20成功 |

## 試作と確認範囲

`.worktrees/body-fitness-direct-model/` に `cfg(test)` の新module、独立f64参照、取得器を追加した。v1の数値コード・保存入力・凍結binaryは保持した。基準936ファイルのhash照合では、既存の `life/mod.rs` とv1のruntime検査入口以外に基準ファイルの差分はなかった。新しいv2ファイルと必要な参照元を[実装目録v2](../../target/body-fitness-partial-groups-20260929/implementation-manifest-v2.json)、実行binaryを[取得固定v2](../../target/body-fitness-partial-groups-20260929/acquisition-freeze-v2.json)に記録した。

独立参照はproducerの群・W/V・prefix sumを使わず、laneから群を作り直し、motion標本を直接列挙する。さらにHarmonic減衰を数値積分し、身体power、source seedと8 sample更新、同Hz分割、帯域端、弱群再配分、不正入力、失敗後状態、scan長境界を確認した。初回unitの2件は参照側の帯域端丸めと、f32量子化後も厳密な同距離になるという誤った期待で失敗した。修正前の結果を保存し、最終debug/releaseは10成功・2取得用ignoreとなった。[debug記録](../../target/body-fitness-partial-groups-20260929/unit-final-debug.log)、[release記録](../../target/body-fitness-partial-groups-20260929/unit-release-rev2.log)を参照する。

整形と標準Clippyは成功。テストを含むClippyには継承元にも存在する27件の指摘が残り、新規moduleの指摘は0件。隔離版の `test_report.txt` と `test_status.txt` は今回の対象unitに更新し、対象範囲をstatusにも記した。v1時点の全体1407成功は保存済みだが、v2を含む全体回帰の成功とは扱わない。

## 旧参照との比較

[全728行](../../target/body-fitness-partial-groups-20260929/semantic-v2.jsonl)を取得し、欠測・重複・unsupportedは0、104群すべて7候補が揃った。[独立読戻し](../../target/body-fitness-partial-groups-20260929/semantic-v2-independent.json)は保存した非正規化bin massとCからscore/levelを再計算し、最大差はそれぞれ約2.71e-8、5.82e-8だった。したがって今回の大きな旧参照差は、出力の集計誤りではない。新式そのものの独立確認はunitの範囲であり、728件すべてのproducerを別実装で再生成したという意味ではない。

最大score差はSine 440 Hz環境、Harmonic dark、基音440 Hz・0 cent。最大level差は同環境、LandscapeDensity、基音440 Hz・24 cent。13 caseすべてに少なくとも1件の誤差上限超過がある。順位逆転3対は厳密な逆転であり、新旧差の符号判定へ同点が混入した件数は0だった。この比較は設計診断にも使った保存入力であり、holdoutでも実runtimeの抽選・占有・親選択検査でもない。

最初の取得器は `landscape_peaks` をModalと仮定して168候補の後で停止した。凍結v1入力の実身体は同case全56行でHarmonicだった。v1単独取得器のModal factory未登録が原因であり、[契約§6](body-fitness-partial-groups-contract-20260929.md#6-取得器の身体種別訂正費用取得前)のとおり入力を変えず期待kindだけを訂正した。途中logと修正前binaryも保存した。完全な728件の取得は修正後binaryによる。

## 費用

690候補＋非格子21候補を全評価し、準備を含む5回の最大値を保存した。上限は1体1 ms、16体8 ms。[全記録](../../target/body-fitness-partial-groups-20260929/cost-v2.jsonl)と[独立集計](../../target/body-fitness-partial-groups-20260929/cost-v2-independent.json)は全20条件＋最大容量観察1条件で一致した。計算エラー0、各条件内のchecksumは5回一致した。取得用テストのexit 101は登録費用条件の失敗を表すもので、途中打切りではない。

| 身体・負荷 | 1体の最大 ms | 16体の最大 ms | 両条件の判定 |
|---|---:|---:|---|
| Sine、K16/U1入力 | 0.076 | 1.000 | 成功 |
| Harmonic、K16/U1 | 0.277 | 4.199 | 成功 |
| Modal、K16/U1 | 0.275 | 4.243 | 成功 |
| Sine、K64/U9入力（実laneは1） | 0.070 | 0.978 | 成功 |
| Harmonic、K64/U9 | 4.126 | 66.800 | 不合格 |
| Modal、K64/U9 | 3.875 | 61.045 | 不合格 |
| Harmonic motion=0.9、K16/U1 | 0.489 | 7.437 | 成功 |
| Harmonic motion=0.9、K64/U9 | 4.392 | 69.978 | 不合格 |
| LandscapeDensity、Harmonic実生成 | 3.499 | 55.058 | 不合格 |
| LandscapePeaks、Modal実生成 | 1.535 | 22.623 | 不合格 |

Kは指定部分音数、Uは指定unison数。最大容量B=2048、N=2048、J=256、K64/U9の1体は12.578 msだった。この別観察には事前上限がないので合否を付けない。

固定・motion条件のCは非平坦な `sin(0.037*b)`。v1の平坦Cとは異なるため厳密な速度比を主張しない。LandscapePeaksは通常runtimeと同じfactory登録を計時前に行い、各候補の実身体kindがModalであることを確認した。v1の未登録fallbackとの同一入力比較にもならない。

身体・scratch・共通環境の準備を含み、16体で共通環境を共有した。Landscapeでは候補ごとのVoice生成も含む。現在の環境を作るPCM/NSGT処理、共存32 Voiceの通常処理、実deviceはこの局所計時に含まない。host専有は保証せず、前後の負荷を記録した。今回の未達を負荷由来と決めつけて再測定しない。

## 近似の境界と次の判断

名目帯域外のlaneを先に除外する定義には非対称性がある。たとえば分析下限100 Hz、名目99 Hz、offset比0.02の単一laneは、その時点の周波数が100.98 Hzでも本モデルでは除外されたままとなる。反対に名目101 Hzが同じ大きさの負offsetで98.98 Hzへ動けば、その標本は帯域外として除く。これは定義から導く反例であり、実PCMを追加取得した結果ではない。名目帯域内重心の丸め補正と、このモデル上の欠落を混同しない。

実backendにある極小motion角の無効化、ADSR、位相干渉、解析窓等は再現していない。新式の内部一致を、実音との一致へ読み替えない。

[保存分布の監査](../../target/body-fitness-partial-groups-20260929/semantic-failure-audit.md)では、v1から45候補が回復し47候補が新たに失敗した。motionの失敗は17→5へ減った一方、motion=0の672候補では140→154へ増えた。最大誤差例では、旧分布が880 Hzへ約0.402の正規化massを置き、その位置のCは1だった。v2ではそのmassがなく、約1077–1085 HzのCがほぼ0の領域へ約0.472が集まった。保存分布の内積でscore差を再現できる。群内部のtraceは保存していないため、弱群再配分等の個別寄与の同定とは扱わない。

続いて[最大例の群形成を独立再構成](../../target/body-fitness-partial-groups-20260929/worst-group-trace.md)した。凍結v2出力に対する正規化massのL1差4.47e-8、score差4.10e-9を確認した上で中間値を調べると、1320・1760・2200・2640 Hzの四群が5%power条件で落ち、すべて880 Hz群へ移されていた。その群のERB重心は1078.769 Hzとなる。したがって、この例で880 Hzの評価massが消えるv2内部の経路は特定できた。旧PCMとの差全体の寄与率を分解した結果でも、全159失敗の原因を同定した結果でもない。

[静的仕事量監査](../../target/body-fitness-partial-groups-20260929/work-bound-audit.md)では、群化後が32以下でも、K64/U9の16体で最大6,552,576回のlane訪問が残ることを確認した。Landscapeは11,376回の候補別Voice生成も含む。契約§4で意図したpower順位の身体ごとの準備は現実装では未達であり、毎候補sortしている。候補依存のModal clamp・同Hz合算等を含めた順位共有の成立条件が未解決である。実測時間の支配割合は、この静的監査からは決められない。

次案では、弱い遠方成分を保持群へ移して重心を変える規則を見直し、実在成分の周波数支持をどう保つかを先に定義する。計算量の削減では群化後の数だけでなく、その前の入力処理と候補依存生成を扱う必要がある。追加の閾値緩和、対象負荷の縮小、同じ取得の反復を次手にしない。同hop出生の統合、全hop・実device検証、採用判断は未完である。

## 採用済み音色計画との境界と採点位置の追加監査

同日の[音色計画D6–D8](../superpowers/plans/2026-09-29-timbre-synthesis.md)を適用し、footprintの正本は代表描画とする。v2は描画との照合を通過していない近道の候補であり、内部式の成功を独立した作者規則の採用へ読み替えない。取得前契約、凍結ソース・binary、159/728の誤差超過と費用10/20成功の結果は変更しない。既存の試作にある身体種別分岐は保存し、新しい生態系側分岐や指定消費者の内部値読取を増やさない。身体モジュールへの既存経路の移設は音色Phase 3の工程に残す。

[旧参照の採点位置監査](../../target/body-fitness-support-20260929/reference-semantics-audit.md)で、`peak_extraction` は重心 `u_erb` と局所極大の `bin_idx` を別々に返し、`SpectralFrontEnd` は後者でA-weightingと主観強度の配置を行うと確認した。v2は重心を配置位置に用いるため、旧参照の弱群再配分を借りても採点位置の意味が異なる。群power等を固定してreadoutだけを選択済みanchorへ戻す単一介入を次の診断候補とする。全partialの質量を元の位置に保つ保証でも、干渉・時間平均・窓応答の差を解く変更でもない。本追記は静的監査であり、その介入の実装・数値・性能取得は未実施。
