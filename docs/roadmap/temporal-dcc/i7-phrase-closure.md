# I7: フレーズ候補と独立した終止・継続診断

2026-09-13。ユーザーの「全体の何パーセントできたか。次を実行」に基づくI7の実装。
I3〜I6の経路と、それ以前の未commit変更を保持した。技術実装・必要な検査を完了。

## 今回の終了点

実音の音群特徴・articulation・周期／まとまり・記憶照合から、継続するforegroundと
境界候補、独立した終止／継続ordinal headを更新し、通常runtimeのUI／reportへ渡す。
同じ末尾でも発行済みの先行予測が違えば終止headの入力が変わる経路を検査する。
境界確率を終止評定や継続評定の代用品にしない。

これは単一文脈の初期診断版 `phrase-base-i7-v1` である。係数と尺度を明示するが、
適合・校正・T5の認知的採否・可聴性・作者採用は行わない。phraseとsectionの同時beam、
I12a/c/dの双方向結合、全曲処理、生成作用は今回の対象外である。

## foregroundと境界

音群ごとに最大15のknown pathとunknown質量を持つ。最初の有効な音群raw descriptorの
観測開始を、初期foregroundの支持された開始として使う。未知の開始へage=0を補わない。
各pathは一つのforeground clockを持ち、四つの出口を提案する。

| 出口 | 後続状態 | 前のspan |
|---|---|---|
| New | 新ID・新credit、現在の観測更新終端から開始 | unresolved linkとして保持 |
| Overlap | 新しいforegroundの時計を開始 | 元開始とheard endpointを持つright-censored link |
| Reinterpret | 解釈IDだけ変更、元の開始・creditを継承 | 同じ支持identityを保持。新しいoccurrenceを加算しない |
| Inactive | foregroundなし、hazard clockなし | unresolved linkとして保持。終止を宣言しない |

Inactiveへの遷移には観測された低energyが必要であり、欠測では選ばない。
低energy基準はI5と同じglobal RMS referenceの1%。許可されないinactive候補の質量を
他の出口へ再配分せず、unknownへ残す。既知のinactive pathでは、再び観測された非低energyと
先行articulationのattack質量を使って再入場を提案する。開始はその観測hopの開始であり、
欠測内へ遡らない。これはforegroundの新しい取得であり、inactive状態からの境界hazardではない。

出口とstayの候補はparent参照だけで列挙し、質量・parent順・出口順で決定的に選ぶ。
上位15を保持してからlink配列をcopyする。捨てた質量と30秒尺度のunknown leakを別に保持し、
knownだけで再正規化しない。現版のunknownには新しいduration clockを与えず、同じ音群内では
吸収状態として残す。unknown-parentからの文脈に応じたreadmissionは統合beamへの残件である。

各pathのlinkは最大16件、1件128 bytes以下。容量不足では支持の小さいlinkを失い、
同点は古いheard endpointから落とす。失った件数を診断に残す。retirementや容量損失を
期待の解決・closure・記憶強化に変換しない。リンクは元のspanとcreditを参照する初期版であり、
expected-segmentごとのcredit/reinforcementと、以前のoverlap spanへの後続証拠再配分はI8/I12cに残す。

## 26成分の時間依存head

hazardと4出口のsoftmaxには別々の係数を与える。入力は既存の登録26成分
`[1, ln(1+elapsed_sec), standardized v1..v12, missing m1..m12]` を使う。

- v1〜v4: I5の現在のattack／continuation／release／gap。状態ごとにone-hotでheadを評価し、
  その後に状態質量で周辺化する。平均したone-hotを非線形headへ入力しない。I5のunknownは4値をmaskする。
- v5〜v8: 先行0.25秒のrise、decline、flux、energy share。元の物理支持で重み付けし、
  epoch／音群開始でclipした窓の90%未満しか有効でなければmaskする。
- v9: 先行2秒のacoustic grouping admission。整数／word候補のdistinct accentの平均支持の最大値を
  各refreshで得てholdする。取得と帰属の有効時間を90%以上要求し、alphaで重み付けする。
  支持された陽性候補は使い、不完全な陰性探索を既知ゼロにしない。
- v10: I4の最新の適格なordered matchの`ln(1+cost)`。queryの全audio支持から0.5秒未満、
  受領済み、期限内、現group/epoch、非曖昧な結果だけを使う。
- v11/v12: I4の初期照合診断はcorrespondence massとそのentropyを供給していないためmissing。
  候補数やbest costから確率を捏造しない。完全なcorrespondenceによる供給はI8/I12cへ残す。

snapshotの`phrase_values`の先頭4値は共通の値に潰さずmissingのまま置き、
別の`articulation_mass`から状態別の26成分を再構成する。12個のmean/deviationは明示設定、
deviation下限は1e-6。elapsedとmaskは標準化しない。

各観測hop間は直前の文脈を固定し、`softplus(a+b*ln(1+d))`を二点Gauss–Legendreで積分する。
1/2/4/8/16 panelと端点2評価、最大64評価、許容差`1e-9+1e-7*abs(integral)`。
I8着手時に四階微分のBernstein上限と丸め余裕を追加し、panel倍増の差だけで収束を判定しない。
旧版は`a=20, b=-40, d=0..1200秒`の冒頭に集中するhazardを見逃した。修正前にRust回帰テストの
失敗を確認し、共有`hazard.rs`へ修正を移した。600条件でPython参照の値と支持／未解決判定を照合した。
省略された観測区間を一つの観測hopとして扱わない。欠落時間には同じ文脈のsurvival priorを
進め、exit質量をunknownへ送り、四出口のラベルは付けない。非収束もunknownに残す。

各groupの1秒先予測は発行時に文脈とpath質量を固定する。`survival + sum(exits) + unknown = 1`。
元支持、availability、発行時刻、horizonを保存する。現版はgroupごとの予測であり、
全strandのfirst-eventを統合した校正済みmixture boundaryを主張しない。

## I4からの発行済み予測と終止head

I4の既存DTWの最後の対応点から、参照episodeの次の一knotを選び、対応点とtempo比から
その到来区間を計算する。新しいDTWは呼ばない。pitch/tempo変換、次区間の90%支持、
非曖昧性、発行時より前の完全な元支持を要求する。期待区間が既に過ぎている場合は発行しない。
値、尺度、元支持、episode/query/group、予測区間をquery発行時にcopyして固定する。

consumerは受領時にもgroup・epoch・query・発行順序・availabilityを確認する。
受領が次hopになっても、予測を発行した後の観測区間だけを照合対象にできる。
後の観測で期待値を修正してから過去を採点しない。異なるqueryの係数や後の最良候補を
前の予測へ遡及適用しない。

新しく観測されたraw descriptorと期待区間の重なりについて、共通の有効10座標の
標準化二乗残差を平均する。先行2秒で実際に照合できた支持により重み付けする。
期待区間以外の音、欠測、未知座標は比較回数にも残差にも加算しない。
終止headは、この発行済み予測の残差を実際に消費する独立のordinal regressionである。

登録closure mapの10値のうち、初期版が供給するのはこの残差だけである。競合内容との相対score、
最終強化からの時間、availability低下、未照合期待支持、path-localなrelease/continuation占有、
R/H/Cの各値はmissingとする。group平均や無音を代用してcompletionを作らない。
これはfull closure mapの適合・回収ではなく、初期の順序文脈接続である。
残りの期待・availability入力はI8/I12c、path-local占有と音響文脈はI12aの統合に割り当てる。

継続ordinal headは別係数を持つ。closure mapに1秒到来確率、pathごとの1秒phrase survival、
先行2秒のaccent density、grouping supportを追加する。I6がちょうど1秒horizonの一点予測を
発行しているときだけ到来成分を使い、別horizonやreset-unknown区間から補間しない。
accent密度は観測時間で割り、128件容量による欠落が対象窓に入ればmaskする。
時間hazardのsurvivalと、出口typeがunsupportedであることは別に扱う。

両ordinal headは5カテゴリ。`P(Y<=k)=sigmoid(cutpoint_k-dot(coefficients,x))`とする。
カテゴリ平均を0〜1へ変換した値は評定期待値であり、無境界確率ではない。
入力は共通の29要素収納形式`[1, standardized 14 values, 14 masks]`。
closureでは末尾4値を使わず、対応する係数8個をゼロに制限するので、継続専用入力から独立する。

group/pathごとの出力分布を、先行2秒の取得されたacoustic assignment支持で混合する。
残差group、失われたgroup/history、known path以外の質量を他groupへ正規化し直さない。
観測窓のcoverageを別に報告し、supported massとの積をrとして、
`reported = r*supported_distribution + (1-r)*explicit_prior`とする。全入力missingのheadは
supported distributionを供給せず、priorへ戻る。T=1の初期診断であり、priorもfit済みであるとは扱わない。

## 操作・通常runtime・資源

`temporal_phrase`は省略時無効。`temporal_mode("observe")`と、I4〜I6の明示設定を必要とする。
既存の設定へ加える例。ここでの係数とpriorは検査用の値であり、採用済みモデルではない。

```python
import json

def array(name, values):
    print(name + ' = ' + json.dumps(values))

print('[temporal_phrase]')
array('means', [0.0] * 12)
array('deviations', [1.0] * 12)
array('hazard', [-3.0] + [0.0] * 25)
array('exits', [[0.0] * 26 for _ in range(4)])
for head in ('closure', 'continuation'):
    print('[temporal_phrase.' + head + ']')
    array('means', [0.0] * 14)
    array('deviations', [1.0] * 14)
    coefficients = [0.0] * 29
    coefficients[1 if head == 'closure' else 12] = -2.0 if head == 'closure' else 2.0
    array('coefficients', coefficients)
    array('cutpoints', [-1.5, -0.5, 0.5, 1.5])
    array('prior', [0.2] * 5)
```

起動前に有限係数、非負deviation、増加するcutpoint、正で和が1のprior、closureの禁止係数を検査する。
観測packet版は7。`phrase_parameters`／`phrase`／`phrase_error`を共通snapshotへ追加した。
Scenarioのsectionラベル・Voice identity・EOFを知覚ラベルとして入力しない。
同じsnapshotをUIとreportへ渡し、未校正と表示する。UIは第二の設定元ではない。

欠測／epoch／退役は観測workerの元の時計に従う。EOFはcensoringであり、最後まで観測した
場合の終止評定を変えない。未配送tailでは既知の時計だけを進め、失った支持からhead出力を作らない。
未配送の新epochで終わった場合は前epochのphrase診断を破棄する。

Phraseの本体をheap所有にして、観測workerの大きな値コピーによるstack overflowを解消した。
threadのstack上限を引き上げて回避してはいない。historyは2秒窓、accentは128件、pathは15件、
linkは16件で有界。観測worker側の処理であり、音声callbackへ照合やphrase更新を追加しない。
全処理・最大候補・二bus/64 Voiceの資源合格はR2/A3で別に判定する。

## 検査

全テスト941成功・0失敗・28ignore。`cargo test exit=0 @ 2026-09-13T10:07:43+09:00`。
通常Clippy・全target check・format/diff検査も通過。個別に以下を確認した。

- 26成分の順序・maskと状態別のhead評価、解析解／細かい積分との一致、長いgapの計算上限。
- 高い境界確率と低い終止評定、その逆の両方。全missing時の明示prior。
- 同じ末尾と異なる発行済み期待値、元支持、遅延配送、未来／異group／古いqueryの拒否。
- 四出口の後続時計・credit、欠測中の出口ラベル不生成、inactive再入場、16 link上限。
- I4の通常照合を使う後続予測: 36件を発行、後続支持を41回照合。
- 実波形→NSGT→観測worker→記憶予測→phrase/closure: 900 hopで後続支持を30回照合。
  終止／継続headのsupported massは約0.285。未校正の尺度による接続検査である。
- offからI7まで9構成×report有無の18条件で、レンダー音声の完全一致・両bus終了・共通診断を確認。
- EOF・未配送epochの既存回帰もI7を有効にして検査。

全テストは`RUST_BACKTRACE=1 cargo test -- --nocapture`で実行し、`test_report.txt`と
`test_status.txt`へ保存する。通常Clippy、全target check、format/diff検査を行う。

次の実装単位はI8「sectionと帰還」。I7の初期接続を、full T5モデル・統合beam・認知的妥当性・
作者採用の合格に拡張しない。

検証版識別: `src/**/*.rs`、`tests/**/*.rs`、`Cargo.toml`、`Cargo.lock`の179ファイル。
相対path順に`path + NUL + file_sha256 + LF`を連結したSHA-256:
`cc69ab6454c57697fc51e53342e80b3149c9a1f66e2809be1759d08a570daeb4`。このhashは未commitの統合作業ツリーに対応する。

## I8着手後の受け渡し更新

2026-09-13、snapshot version 2/observer packet 8へ更新。親経路、閉じた区間の元支持、
元cutで凍結した末尾6特徴を追加した。accent密度をadmission重みの和で計算する。
I8の受け渡し記録に検証を集約する。全Rust964成功・0失敗・28ignore、通常Clippy・全target check通過。
