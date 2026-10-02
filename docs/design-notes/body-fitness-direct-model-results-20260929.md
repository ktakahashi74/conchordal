# 直接部分音モデルv1の検証記録

2026-09-29。[事前契約](body-fitness-direct-model-contract-20260929.md)に基づく隔離試作。主計画は[身体適応度計画](body-aware-fitness-plan.md)。本書は取得済み結果と未完の検査を分けて記録する。

## 現在の結論

**v1は意味判定・費用判定とも不合格。本番接続へ進めない。** 直接部分音という設計方向全体の反証ではなく、現在の静的power代理分布が登録した範囲では旧参照に十分近くなく、全候補を走査する費用も対象負荷の条件を満たさないという結果である。

| 項目 | 結果 | 事前条件 |
|---|---:|---:|
| score最大絶対差 | 0.4474703673 | 0.025以下 |
| level最大絶対差 | 0.1849887030 | 0.0125以下 |
| 候補ごとの未達 | 157 / 728 | 全候補通過 |
| 旧score差0.1以上の候補対の順位逆転 | 3対 | 0対 |
| score最大候補の変更 | 16 / 104組 | 診断値 |
| fixtureの欠測・余分・重複 | 各0 | 各0 |

最大差は他者Sine 440 Hz環境におけるmotion付きHarmonic、基音440 Hz、0 cent。Rust記録と独立f64内積の差は丸め程度であり、集計ミスではない。全728件、環境4件、選択群104件、summary1件を検算した。機械集計のfailed rows=158は候補157件とsummary1件の和であり、候補数と混ぜない。

記録は[JSONL](../../target/body-fitness-direct-model-20260929/semantic-v1.jsonl)、[独立検算](../../target/body-fitness-direct-model-20260929/semantic-v1-independent.md)、[機械可読検算](../../target/body-fitness-direct-model-20260929/semantic-v1-independent.json)。選択用の環境density_mass_effが保存行にないため、抽選weights自体の独立導出はしていない。実runtimeの占有・spacing・乱数消費・親選択を検証した結果でもない。

## 試作とコード検証の範囲

`.worktrees/body-fitness-direct-model/` にtime4版を固定コピーし、`cfg(test)` の直接モデルと検査入口だけを追加した。mainの実消費者には接続していない。入力、固定身体と候補依存Landscape身体の区別、上限、失敗後のscratch消去を試作で定義した。[実装目録v2](../../target/body-fitness-direct-model-20260929/implementation-manifest-v2.json)と[取得時のbinary・入力固定](../../target/body-fitness-direct-model-20260929/acquisition-freeze-v1.json)を保存した。

新式の独立f64式、入力・容量・scan境界、同bin合算、帯域外、Modalの高域clamp、unison、振幅scale、失敗後の状態について、最新版のrelease単体検査は5件成功、取得用3件は通常実行時ignore。これは新式の実装確認であり、旧参照との意味判定の合格ではない。

その後、新規試験の引数整理とrelease専用ガードのlint表記を修正した。数値式・fixture・判定値は不変。現sourceは[実装目録v3](../../target/body-fitness-direct-model-20260929/implementation-manifest-v3.json)、最新debug単体検査は5件成功・3件ignore、`cargo fmt --check` と標準 `cargo clippy -- -D warnings` は成功。`cargo clippy --all-targets -- -D warnings` は継承した試験等の27件で失敗し、新しい直接モデルの指摘は0件となった。[旧sourceとの照合](../../target/body-fitness-direct-model-20260929/clippy-baseline-attribution.json)で全27件の該当行が継承元にも存在することを確認した。今回の対象外を一括修正していない。数値・費用取得用のv1 binaryは変更せず、hash一致を再確認した。

最初の全体 `cargo test` は旧非同期action recoveryの30秒wall-clock待ちで1件失敗（lib 1298成功・1失敗・58 ignore）。同じ失敗テストの単独再実行は2.79秒で成功した。新直接モデルはその失敗経路から呼ばれない。競合負荷の影響という仮説はあるが、原因同定とはしない。閾値や旧テストは変更していない。

並列数4の全体再検査は12:09:09 JSTにexit 0で終了した。lib・integration等の43結果群を合計して1407成功・0失敗・58 ignore。旧factorialの8 renderとrespawn/selectionも含む。[全体log](../../target/body-fitness-direct-model-20260929/cargo-test-threads4.log)、[終了記録](../../target/body-fitness-direct-model-20260929/cargo-test-threads4.status)、[集計](../../target/body-fitness-direct-model-20260929/cargo-test-threads4.summary.json)を固定保存した。隔離版の `test_report.txt` と `test_status.txt` は、その後の試作に伴う最新の検査で更新する。この全体binaryは追加境界検査前の版であり、最新版の追加境界・lint修正は上記単体結果で別に確認した。全体回帰の成功と、新モデルの意味・資源判定の失敗は別の事実である。

## 原因の切り分けと資源検査

[失敗診断](../../target/body-fitness-direct-model-20260929/semantic-failure-audit.md)では、motion例の周波数方向の広がり、高次部分音の減衰、既存peak抽出の弱成分処理、近接unisonの干渉・統合を別の差として挙げた。保存分布だけで各過程の寄与率は同定できない。

[Harmonic減衰の単一介入診断](body-fitness-direct-damping-diagnostic-20260929.md)も完了した。実backendの定数から求める係数だけを追加し、同じ保存728件で比較した。介入前の独立v1再構成と解析積分の検算は通過した。平均score誤差は0.021557から0.019047へ縮んだが、最大誤差は0.447470から0.474818へ増え、未達候補も157から160へ増えた。明確な順位逆転3対、最大候補変更16群は変わらなかった。[介入結果](../../target/body-fitness-direct-model-20260929/damping-diagnostic-v1.md)を保存した。減衰だけの追加では進行条件を満たさず、motion・位相・peak処理等の残差が未説明である。

[受入条件の意味監査](../../target/body-fitness-direct-model-20260929/acceptance-semantics-audit.md)では、0.025等は旧72-frameとの近接性を調べる事前screenであり、出生の音楽的妥当性や実選択への許容差としては未導出と確認した。特に基準の由来である0.1は移動greedyの閾値で、出生の抽選・最低level・親選択を直接規定しない。今回の不合格はそのまま保持するが、別定義の身体表現そのものが誤り、あるいは出生が実行不能とまでは結論しない。以降の設計では、旧解析モデルへの近似と、新しい身体表現として守る機能を区別する。現在の定義・閾値ではruntime接続への事前条件を満たさない。

v1の冷状態費用screenも完了した。全体回帰と自分のbuild/lintの終了後、凍結したv1 binaryで登録済みの固定身体12条件、候補依存Landscape生成4条件、最大容量の別観察1条件を逐次取得した。全17件・各5標本に欠測や重複はない。CLIはexit 0、取得用テスト2件は正常終了したが、**費用gateは0/16成功**。機械可読記録は[JSONL](../../target/body-fitness-direct-model-20260929/cost-v1.jsonl)、検算は[独立費用監査](../../target/body-fitness-direct-model-20260929/cost-v1-independent.md)。

| 690候補＋非格子21候補、準備込み | 1体の5回最大 ms | 16体の5回最大 ms |
|---|---:|---:|
| Sine、K16/U1入力 | 1.426 | 24.316 |
| Harmonic、K16/U1 | 1.482 | 23.693 |
| Modal、K16/U1 | 1.485 | 23.704 |
| Sine、K64/U9入力（実laneは1） | 1.384 | 22.209 |
| Harmonic、K64/U9 | 2.995 | 49.271 |
| Modal、K64/U9 | 3.039 | 48.832 |
| LandscapeDensity、候補ごとにHarmonic身体生成 | 4.545 | 72.908 |
| LandscapePeaks、候補ごとにModal指定で身体生成（下記注記） | 2.641 | 40.907 |

事前上限は1体1 ms、16体8 ms。16体の費用は10.667 msのhop全体も超えており、単なる微小差の未達ではない。最大容量B=2048、N=2048、J=256、K64/U9の1体は19.636 msだったが、この別観察には事前上限がないので合否を付けない。hostの専有は保証しておらず、取得前後の負荷は記録した。負荷を理由とする再測定は行っていない。

固定身体screenの環境Cは全bin 0.2なので、checksumの一致は非平坦な場での選択を検査しない。Landscape screenは実環境を使う。SineのK64/U9入力を576 lane負荷とは数えない。取得値はこのbinaryと入力の局所費用であり、全hop・共存32 Voice・deviceの測定ではない。

v2取得器の入力確認で、v1の `landscape_peaks` semantic 56行は、Modal指定に対して実BodySnapshotが全件Harmonicだったと判明した。単独取得器が `life::modal::register_modal()` を呼ばず、初期registryにModal factoryがないため、生成経路がHarmonicへfallbackしていた。費用取得器もこの登録を呼んでおらず、上表のLandscapePeaks 2条件を実Modalの費用証拠として扱えない。保存値と不合格を保持し、意図したModal条件の被覆不足を加記する。直接BodySnapshotを渡した固定Modalの単体・semantic・費用条件は、このfactory生成経路を通らない。

## 次方式の静的調査

[相対powerの再利用監査](../../target/body-fitness-direct-model-20260929/relative-power-factorization-audit.md)では、固定snapshotのlane別ratio/powerは準備を共有できる一方、絶対Hzでの聴感補正、Nyquist、Modalのclamp、候補依存Landscape生成を候補ごとに扱う必要があると確認した。実ToneのHarmonic motionは、部分音比に比例した個別vibratoではなく、全laneへ `f0*(vibrato+jitter)` を足す共通Hz変動である。旧 `HarmonicBody` の別経路を代表Toneのモデルへ転用しない。

[実kernel窓の解析応答案](../../target/body-fitness-direct-model-20260929/direct-spectral-front-end-design-audit.md)も調べた。PCMを省いても全候補×72時点×lane×bandの複素応答を計算する構成は大きな仕事量を再導入し、静的Sine/Harmonicだけではmotion/Modalを覆わない。現段階では本番候補として実装しない。単純な疎powerを旧frontendへ通すだけで旧peak処理を忠実に再現できるとも扱わない。次の方式は、身体表現の意味と費用上限を同時に定義してから選ぶ。
