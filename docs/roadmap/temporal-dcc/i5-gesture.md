# I5: articulation候補と音群間gestureの診断

2026-09-13。ユーザーの「I5を完了させて」に基づく実装。対象はI5の技術接続であり、
T1の認知的妥当性・可聴性・作者採用、適合済みモデル、I12のphraseとの双方向結合は含まない。
I3・I4の既存変更を保持し、その上に通常runtimeの観測経路を追加した。未commitの作業ツリーが対象。

## 入力と状態更新

`temporal_mode("observe")`、既存の`temporal_ridge`／`temporal_acoustic`に加えて
`temporal_gesture`を明示した場合に、両busへ独立した`gesture::Gesture`を構築する。
入力は実音のNSGTを通った音群raw descriptor、更新前のfractional assignment、現在のridgeである。
Voice ID・Scenarioのsectionラベル・作者のgestureラベルは入力しない。I4の固定長診断区間も
gesture境界として流用しない。I4の検索出力を使うphrase文脈は今回のbase経路に不要で、後続に残す。

現版は単一文脈のbase articulation投影である。attack／continuation／release／gapの候補を
最大15経路と一つのunknownへ保持する。各候補に状態への入場時刻、attack・release・観測gapの
区間と元NSGT支持・availabilityを保持する。同じ経路をcoalesceした後に質量順で15件へ絞り、
捨てた質量をunknownへ送る。時刻・状態・runによる固定tie規則を使い、残った候補だけで再正規化しない。

既存仕様§9.2の競合率を実装した。対角を除く各遷移のrateは`softplus(theta_sr · x)`、
stayは`exp(-sum(rate)*dt)`、各exitは`(1-stay)*rate/sum(rate)`とする。
新しいgapだけを、音声と音群への所属が観測され、そのRMSが明示したglobal development referenceの1%以下の場合に許す。
referenceはfull-scale amplitude単位で下限1e-6。現在のbusや音群からreferenceを推定し直さない。
既知gapのstayは欠測中も事前として残り得るが、欠測から新たなgapを作らない。

I5のbase入力layout `articulation-base-i5-v1`は次の11成分で固定した。

1. intercept = 1
2. raw descriptorのlog-RMS上昇
3. 同下降
4. 同spectral flux
5. 現在のridgeの第1傾き候補の絶対値を、該当音群へのfractional assignmentで重み付き平均したmotion
6. `log(1 + state_elapsed_sec)`
7. 上記五値のmissing indicator（同じ順序）

motionは有効なridge slopeがある所属だけを分母へ使い、分母ゼロはmissingとする。
五値を明示したglobal mean／deviationで標準化し、deviation下限1e-6、欠測値は標準化後ゼロとする。
coefficientの添字は[from state][to state][11 coordinates]、状態順はattack、continuation、release、gap。
対角coefficientは使わない。これは初期の数値入力定義であり、fit済みの知覚的尺度ではない。
実装・fixtureではphrase項と観測log-potentialを中立に保ち、stage 2のbase rateだけを接続する。
既定値としてのfitや新しい聴取ラベルを導入していない。

許された既知遷移の数値重みが全てゼロの場合は全質量をunknownへ送る。
既知候補の遷移を許された状態間で正規化した後、`eta=1-exp(-dt/10 s)`をunknownへ割り当てる。
unknownからは新しい観測支持がある場合だけ、許された既知状態へ一様なraw admissionを作る。
この10秒は登録済みの初期工学尺度であり、認知的な保持時間ではない。
欠測をまたいだrunの端点はcensoredにし、観測した新しいattackから新runを始める。
出生・split・merge・世代変更では他の音群のrunや取得支持を継承しない。

## 音群間の投影

0.1秒ごとに、最大8個のsingle family（残差を含む）と、最大7 resolved groupの21 pair familyを評価する。
支持窓は現在時刻から8秒を各groupの生存区間でclipしたもの。各hopで
`alpha_g = sum_i a_ig / sum_h sum_i a_ih`を積分してBを求め、八groupのB合計でbへ正規化する。
追加の音量・salience重みを掛けない。分母ゼロはunsupported。
coverageは同じ生存区間内の有効な音声・所属の物理的時間割合とする。

singleのraw priorはb、pairはsqrt(b_g*b_h)。全familyで正規化してからcoverage>=0.9を適用する。
不足するmemberを含むfamilyの重みを他へ配り直さない。family内は保存経路のq、pairではq_g*q_hを使う。
resolved pairはattack順に比較し、先行runのrelease支持、release開始の2秒前以降というlead条件、
観測したcontinuation／releaseとの重なり、または観測gap開始後1秒以内という条件を満たす場合にunionを作る。
release開始をgap開始の代用にしない。端点の区間が判定境界を跨ぐ場合、または欠測・censoringがある場合は
unresolvedへ残す。等しいattack開始、releaseを伴わない持続の重なり、条件を外れた確定pairはsingleへ分ける。
分ける重みはb_g/(b_g+b_h)、b_h/(b_g+b_h)。現在のnon-gap run、または直前8秒以内の最後のrunだけを対象とする。

同じmember/runを持つ候補の質量を足してから上位16件を保持する。未解決・mask・切り捨てを合わせた質量を
別に残し、常に`sum(retained mass)+unresolved=1`とする。incoming／outgoing union massはこのrefresh結果から
診断へ出す。次のstage 3で使う際は前refreshの値を入力する。現版は生成やphraseへ作用させない。

候補payloadはrunの複製ではなくgroupとrun indexを参照し、96 bytes以下。
単一文脈での最大10,880件のscratchを構築時に確保して再利用する。現在の15 known path構成では
最大9,570件を投影前に列挙する。診断snapshotには上位候補が参照する最大32件のrunの証拠を別にコピーする。
これはsource groupingの合流でも、三音群以上を直接表すgesture partitionでもない。
三音群以上の直接unionは表現範囲外であり、T1の表現漏れ判定を免除しない。

## 操作と診断

`temporal_gesture`を省略すると無効。既存の音群設定を使い、追加設定だけを次のPythonで生成できる。
これはstage 1で規定されたoff-diagonal 1/(3 s)のbootstrapであり、採用済みfitではない。
出力を既存の診断用TOMLへ加える。巨大なcoefficient配列を手で列挙しないための生成例である。

```python
import json, math
intercept = math.log(math.expm1(1 / 3))
coefficients = [[[intercept] + [0.0] * 10 for _ in range(4)] for _ in range(4)]
print('[temporal_gesture]')
print('rms_reference = 0.1')  # Explicit assay reference, not fitted development statistics.
print('means = [0.0, 0.0, 0.0, 0.0, 0.0]')
print('deviations = [1.0, 1.0, 1.0, 1.0, 1.0]')
print('coefficients = ' + json.dumps(coefficients))
```

UIは各groupの4状態とunknown、gesture候補数、二音群union数、unresolvedを未校正と明示する。
reportは同じsnapshotに加え、候補のrun index、元支持、availability、coverage、窓・生存区間を持つ。
入力契約／数値エラーはそのepochの診断を停止して隠し、理由を出す。epoch変更で再構築する。
EOFには最後の受信支持ではなく入力終端を使い、未配送の新epochなら旧診断を破棄する。
古い観測支持をEOFへ付け替えない。

## 検証と終了判定

- `temporal_cognition::gesture::tests`で、競合率の解析的survival、細分した基準との比較、
  観測envelope入力への応答、gap admission、元NSGT支持、欠測時のcensoringを検査する。
- handoffは0.1／1／5秒releaseの後の0.25／0.75秒gapを正のfixture、1.25秒gapを負のfixtureとする。
  観測した重なり、無関係な持続の重なり、等しいattack、未観測gapも分ける。
- family正規化、残差、maskされたmember、singleへの分解・coalescing、29 familyの最大投影、
  上位16件の切り捨て質量、8秒窓のclip、新世代の生存区間を検査する。
- 実際のacoustic frontendから状態・gesture候補を更新する接続検査を行う。
- `tests/render_binary.rs`の観測検査はgesture有無を加えた6条件×report有無の12レンダー。
  全WAVがbyte一致し、実音→NSGT→音群→gesture診断、二busの所有者・質量保存・元支持まで確認する。
- 設定の依存、不正尺度・reference・非有限coefficient、未知キー、TOML往復、EOFの配送欠落を検査する。
- 最終全テスト: **921成功・0失敗・28ignore**。`RUST_BACKTRACE=1 cargo test -- --nocapture`、
  終了コード0、2026-09-13 09:07:52 JST。stdout・stderrを`test_report.txt`、終了コードと時刻を
  `test_status.txt`へ保存した。
- 最終版の通常Clippy（`cargo clippy -- -D warnings`）、`cargo check --all-targets`、
  format検査、`git diff --check`が通過。I5の技術接続と検証の終了条件を満たした。

### 実音での二音群union到達

別の3.1秒レンダーでも二音群unionを確認した。上記bootstrapから
`coefficients[0][2][2] = coefficients[1][2][2] = 100.0`だけを変更し、
観測したlog-RMS下降へのrelease遷移応答を強めた接続検査である。適合済みcoefficientではない。
音群設定は[I3・I4記録](i3-i4-observation-memory.md)と同じ、sample rate 48,000、DCC coupling 0を使う。

```rhai
temporal_mode("observe");
seed(9132026);
let a=place(sine().sustain().anchor().amp(0.1).send(presentation_bus), at(220.0));
wait(1.0);
let b=place(sine().sustain().anchor().amp(0.1).send(presentation_bus), at(440.0));
wait(1.0);
release(a);
wait(0.7);
release(b);
wait(0.4);
```

`conchordal-render`でWAVとreportを生成し、終了コード0、観測record 62件、gesture error 0件。
presentationの二音群unionを7 recordで確認し、最大の単一候補質量は
0.00011373235603080442だった。habitatにはunionなし。異なる音群のrun、先行release、
後続attack、censoringなし、元availabilityが観測cut以前であることをreport上で照合した。
質量は小さく、候補への到達を示す結果である。聴覚上のhandoffを支持する確率や認知検証の成功ではない。

再現入力のSHA-256:

- Rhai: `2d240edda185b2f570234db74a70d382a8f1768dcf1feef74a015764bc9f0879`
- TOML: `4506a3efd10d0923a751cf27458ed0285ca0a9ae6c12c09afd2e2da121c0b3bb`

制御raw／経路での正例検査と、実音から診断までの到達検査は別の証拠である。
bootstrapの状態重みを聴取者のattack／release判断と一致した結果とは扱わない。
追加の経路、支持履歴、scratch、snapshotコピーは専用observerの費用であり、全worker負荷・実機・
長時間動作・描画の操作確認はR2／A3へ残す。フィット、共有文脈の8候補との統合、T1各gateも未完。
次の実装単位はI6「周期と到来」。今回I6へ自動拡張しない。
