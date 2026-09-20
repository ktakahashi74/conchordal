# I6: 周期・まとまり候補と到来予測の診断

2026-09-13。I5完了後のユーザー指示「次」に基づく、I6の技術実装。
対象は実accentから時刻付き診断までの接続と、小型MR2の交換境界である。
適合・校正・T2/T3の認知課題・可聴性・作者採用・生成作用は後続R/Aに残す。
作業ツリーは未commit。I3〜I5と、それ以前の既存変更を保持している。

## 通常runtimeの経路

`temporal_mode("observe")`と既存の`temporal_ridge`／`temporal_acoustic`に加え、
`temporal_period`を明示すると、各busの観測workerで一つの音群frontendを
`recurrence::Recurrence`が所有する。音声解析や音群検出を二重実行しない。
同じfrontendの出力をI4の記憶とI5のgestureにも渡す。

従来の検査用recurrence・period・grouping部品を通常ビルドへ接続した。
最大7音群それぞれに、32秒・128 accentのledger、241点のlog-period grid
（0.125〜4秒、1/48 octave）、最大8周期候補、最大16まとまり候補を持つ。
まとまりは整数2〜16と、既存の順序付き比率word探索を使い、0.1秒ごとに更新する。
同じ周期の整数候補だけに絞らず、元の時刻・周期・nominal durationと実測durationを区別する。
word symbolは既存の`[1/4, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 4]`への添字である。
残差にもaccent ledgerを持つが、resolved groupの周期・到来headを残差へ流用しない。

accentのcreditは更新前handleへ一度だけ渡し、その後に出生・split・merge・退役を反映する。
新しい音群は空のledger、周期、まとまり、到来状態から始まる。退役群の最終証拠は診断に
残せるが、activeな到来予測として配送しない。容量置換と欠測による時間経過も既存の所有規則を保つ。

## 到来headの入力と時計

`arrival-i6-v1`は登録仕様の18成分を固定した。係数を内部推定せず、設定で明示する。
添字は0始まり。

| 添字 | 入力 |
|---|---|
| 0 | intercept = 1 |
| 1 | `ln(1 + elapsed_sec)` |
| 2〜4 | `ln(I1/I0), ln(I2/I0), ln(I3/I0)`。I0は直近の完了観測interval |
| 5〜7 | 上記3値のmissing indicator |
| 8〜12 | `log2(P_best), grid_support, cos(2πphi), sin(2πphi), word_indicator` |
| 13〜17 | 上記5値のmissing indicator |

8個の特徴値を明示したmeans／deviationsで標準化する。deviationの下限は1e-6。
欠測値は標準化後0、maskは1。interceptとelapsed項はこの標準化の対象外である。
最強周期の同点は短い周期を選ぶ。phaseの原点は最後の実accent。
周期がなければ5値ともmissing。wordは同じ周期・最新accentまでの探索に限り、
未更新、容量による不完全さ、endpoint不足はmissingとする。完了した陰性は既知0と区別する。

起動時にaccentを作らず、releaseや予測した拍もresetにしない。観測された省略ではelapsedと
survivalを進める。取得欠落や音群への帰属不明では、無resetと見えないresetの両方を残し、
欠測末尾でelapsedを`[0, 最後の実accentからの時間]`とする。観測再開後は両端を進める。
次の実accentで再び一点に戻す。欠測をまたぐintervalは比率履歴に入れない。
4 intervalの元支持は、period ledgerの32秒窓より古くても失わない。

遅れて届いたaccentは、その元支持より後の欠測を解消しない。EOFに未配送の時間があれば
時計とexpiryだけを進め、元支持をEOF時刻へ書き換えない。未配送の新epochで終了した場合は
古いepochの診断を破棄する。

## MR2の二つのモデル

共通の入力adapter、音群所有管理、予測payload、配送検査、UI／reportを使い、設定の`model`で交換する。
算法の分岐は到来Engine内に閉じる。双方の予測対象は「発行時刻の後、horizonまでに次のaccentがあるか」。
予測にはmodel・version・音群handle・元支持・availability・発行時刻・horizon終端・最後の実accent・
elapsed範囲を付ける。consumerはモデル、世代、因果順序、期限を検査する。既発行の値は後のaccentで改稿しない。

- `hazard`: `softplus(coefficients · x)`を毎秒のhazardとする。将来horizonではinterval／period文脈を
  固定し、elapsedとphaseを進める。二点Gauss–Legendreのstep doublingで積分する。
  一積分あたり最大62評価。差の許容値は`1e-8 + 1e-6*abs(integral)`。
  位相周期の1/4以下のpanel幅も要求し、粗い周期aliasによる偽収束を避ける。
  上限内で収束しなければ確率を未解決にする。観測hopのsurvival積分も同じ検査を使う。
- `periodic`: 同じ最強周期と最後の実accentを原点にした反復gridの次回時刻を使う対照。
  horizon内の予測到来の有無を0/1で返す。これは決定的なgrid予測であり、校正済み確率ではない。
  省略でも実accent時計をresetしない。欠測時は周期全体を覆うhorizonなら1、それ以外は保守的な`[0,1]`。
  hazard由来のsurvivalはこのモデルには供給せず、`observed_survival=null`とする。

hazardの欠測時出力はelapsedと位相の区間演算による包絡範囲である。位相の内部極値も含める。
reset時刻の事後分布や、隠れたaccentの個数を仮定した平均確率ではない。
積分の評価数は将来horizon分をpayloadへ出し、観測survival分には別途同じ上限を適用する。
大きなhorizonと強い位相依存では未解決が増える。これは上限のある診断実装の制約であり、
係数の適合や最大負荷R2の合格を意味しない。

## 設定と表示

既存の診断用TOMLへ追加する例。hazardは定数2/秒の数値対照であり、fit済み尺度ではない。

```python
import json, math
print('[temporal_period]')
print('model = "hazard"')
print('coefficients = ' + json.dumps([math.log(math.expm1(2))] + [0.0] * 17))
print('means = ' + json.dumps([0.0] * 8))
print('deviations = ' + json.dumps([1.0] * 8))
print('horizon_sec = 0.1')
```

比較時は`model = "periodic"`だけを替える。起動ごとに空状態から始める。
`temporal_period`省略時は無効。horizonは0より大きく32秒以下、係数・平均は有限、
deviationは有限かつ非負を要求する。未知のモデル名・設定キーを拒否する。

観測packet版は6。共通snapshotへ`period_parameters`、`period`、`period_error`を追加した。
各groupの`period_source`は`[source_start, source_end, available]`。
UIは候補数と予測範囲・支持時刻を表示し、reportは同じsnapshotの詳細を保存する。
出力をuncalibratedと表示し、通常の生成作用は有効化しない。

## 検査と証拠

全テスト930成功・0失敗・28ignore。`cargo test exit=0 @ 2026-09-13T09:37:27+09:00`。
通常Clippy、全target check、format、diff検査も通過した。以下の個別検査を含む。

- 定数hazardの解析解、観測省略による時計進行、elapsed依存、18成分layoutとphaseの連続性。
- 欠測後のreset範囲、cross-gap intervalの除外、位相内部の数値積分を含む包絡、上限での未解決。
- 音群frontendから共通MR2 consumerまで、同一入力360 hopで省略・tempo変更・非周期列を比較。
  350組の予測で支持と時刻が一致し、350組でモデル間の値に差。複数周期候補を持つgroup-frameは302。
- 実波形→NSGT→accent→周期／まとまり→到来。480 hopのpulse・silence・steadyと二尺度の6条件。
  明示尺度のpulse条件でhazard予測940件、基準尺度のpulse条件でperiodic予測894件。
  silenceの両条件はaccent・周期・まとまり・予測とも0。
  低thresholdのsteady条件では予測も発生する。この陽性は知覚的な周期回収の証拠として扱わない。
- EOF・未配送epochの観測worker回帰。既存の所有・容量・word順序・非整数まとまり検査も維持。

実レンダーではoff、observe、ridge、acoustic、memory、gesture、hazard、periodicの8構成を
reportあり／なしで比較し、16条件すべてで両busのEOF・共通診断・音声完全一致を確認した。
この短いレンダーの主目的はruntime接続と非干渉であり、周期回収は上のNSGT fixtureで別に検査する。

再実行:

```bash
cargo test --lib temporal_cognition:: -- --nocapture
cargo test --test render_binary temporal_observation_preserves_audio_and_finishes_both_buses -- --nocapture
cargo clippy -- -D warnings
cargo check --all-targets
```

全テストのログは`test_report.txt`、終了状態は`test_status.txt`。両方とも後続作業で更新される。
[小型記録](i6-period-arrival.json)に対象sourceのhashと結果を保存する。
次の技術単位はI7「フレーズと終止」。MR2の再適合・予測安定性O12・T2/T3の採否と、
I12bのgestureとの双方向結合は未完了のまま残す。
