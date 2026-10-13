# 次のセッションの始め方

次の段は「機構の完成」。作者は15:52頃に機構を先、曲を後とし、16:44にT-17の案B（L1〜L4）と、
A-14の形を選べる創始方策として入れる方針を採った。再開は[現行計画§8](plan-current.md#mechanism-completion-scope-20261013)と
[台帳§9.3.81](../design-notes/technote-ledger.ja.md#mechanism-completion-scope-20261013)と
[§9.3.82](../design-notes/technote-ledger.ja.md#pitch-movement-requirement-and-steady-20261013)から始める。
17:07に音高移動側のL5を追加し、分母は五要件となった。17:42にL2の期待の対象とii-Bを採択し、
T-19で試作中。L3の候補AとL5の案Cは設計の推奨で、統括がA-18とA-17の外部試作へ進めた。
作者の設計採択ではない。L5の出来事の契約→L3の消費者の順で接続し、両試作は繋がず別々に音にする。
V-14のbranchは、第四のtimingのpreset `steady()`の本体実装と、V-13の56件およびA-14/P-4の三seedとのWAV・非計時reportの一致を備える。[台帳§9.3.84](../design-notes/technote-ledger.ja.md#steady-private-opportunities-20261013)と機械に固有のV-14の報告・pinから確認する。既定と既存presetは変えず、既存entrained／flowの契約とL1全体の受入は残る。steadyへの到来の作用は未採択で、採った構成の実機はL4で扱う。統合とpushの状態は作業時のgitから別に確認する。
P-4の基準と境界50だけの二本は17:03に試聴済み。境界の省略値0は変えていない。
新しい採択と設計結果は[台帳§9.3.83](../design-notes/technote-ledger.ja.md#l2-expectation-adoption-and-design-results-20261013)と
[再開点](#l2-expectation-adoption-and-design-results-20261013)を読む。以下の過去の結果は取得時点の記録として読む。

2026-10-13更新。現行計画の20単位は2026-10-08にすべて出口に達した。結果とその後の追加作業は[plan-current.md §7](plan-current.md#7-結果)に記録した。T-4は不成立、T-6は到来の重み1の採用と、検出の数値を選定できない問題の名指しによって閉じた。N-7の夜の再測定で昼のB′不合格を解消し、その後、T-7の最終sourceによるA・B・A′・B′の四構成の実機再測定も合格した。過去の不合格は履歴として保持している。

作者は2026-10-10に動的代謝の受入を改訂してN-7を受け入れた。N-7は同日にmainへ統合され、originへpush済みである。時間構造は、原典の図の数値再現を一律の門にした誤りを撤回し、T-6kで整数比の階層の設計を閉じ、T-7で本体に実装した。T-7も同日にmainへ統合し、最終検査の後、11:50にoriginへpushした。作者の試聴を受けたmetric内の位相分離のT-8〜T-10は、午後の範囲の判断で停止した。拍の同期の二つの失敗はT-11で設計の選択と分類した。F1はT-12で設計を本体の外で閉じ、T-13で実装・回帰・文書をbranchへ用意した。その後2026-10-11に作者はF1の採用方針とC＋hardの最小形を決め、T-16がT-13の上に実装を積んだ。F1は実機の四構成合格後、08:28に統括がmainの`c2b8308`へ統合した。その後08:46に、統括はT-16をmainの`000b884`へfast-forwardで統合した。慣れと既存語彙の旋律らしい線を作者が評価し、「動いた時にだけ発音する」V-1は外部試作を終えた。2026-10-11に作者が本体への実装を採り、09:03頃に `pitch_apply_mode("rearticulate")` の形と名前を確定した。統括が併用規則の案Aを採り、本体実装・検査・文書をV-2のbranchに用意した。[午後〜夜の状態](plan-current.md#temporal-melodic-outcomes-20261010)を次の作業の入口にする。20単位の終了、実装・統合・push、作者の音楽的な評価を区別する。

作者は同日、慣れを既定で有効にする判断も採った。[H-3の既定有効化](#habituation-default-branch-20261010)はbranchで用意した後、作者の指示を受けた統括がmainへ統合し、22:57にoriginへpushした。push後のmainとorigin/mainはともに`39e6840`だった。慣れ有効の四構成の実時間は既に合格しており、既存の科学的な対照では無効を明示する。

## 1. 始め方

統括はClaude Opusのセッションが行う。Emacsでこのcheckoutのagent-shell（Claude）を開き、§2の文書を読んで現状を確認する。新しい単位は、必要な作者判断の後に、作業範囲と出口を明記して割り当てる。2026-10-07の段0・段1を再実行する指示は使わない。

workerは `gpt-6.1-sol` の `xhigh`、KEIOアカウントのagent-shell bufferである。統括が `scripts/worker-shell.el` で起動、指示、状態確認を行う。以下の `T-1` は起動構文の例であり、済んだ単位の再実行指示ではない。

```sh
# 起動（返り値はbuffer名 "Codex worker T-1 @ conchordal"）
agent-emacsclient --eval '(progn (load "/home/shafi/lwrk/conchordal/scripts/worker-shell.el" nil t) (conchordal-worker-start "T-1"))'

# 指示。briefはfileに書き、送る文は短くする
agent-emacsclient --eval '(conchordal-worker-send "Codex worker T-1 @ conchordal" ".orchestration/units/T-1/brief.md を読み、その単位を実行せよ。")'

# 状態。各workerの (名前 アカウント 実行中か) を返す
agent-emacsclient --eval '(conchordal-worker-status)'
```

- モデルと努力度は、このEmacsのCodexの既定（`gpt-6.1-sol`、`reasoning_effort` = `xhigh`）から入る。アカウントは `scripts/worker-shell.el` がKEIOを指定する。指定しないと、このcheckoutのCodexはRIKENになる。
- 2026-10-07の試験起動で、KEIO（`CODEX_HOME=~/.codex-keio`）、`gpt-6.1-sol`、`xhigh` と、作者の表示中のwindowを変えないことを確認した。過去の試験・統括bufferが今も残っているとは扱わず、起動前に状態を確認する。
- workerの結果は、briefで指定したfileに書かせる。統括はそのfileを読む。完了の検知は `conchordal-worker-status` の「実行中か」と、結果fileの有無で行う。

## 2. 先に読むもの

1. [plan-current §8「機構の完成」](plan-current.md#mechanism-completion-scope-20261013)：採択した五要件の出口、依存、分母に入れないもの、記録時点の状態。
2. 機械に固有の`.orchestration/units/T-17/report.md` §§2〜6：本体のM01〜M17、残るK1〜K11、案Bと公開の判定の関係、統括の要約の訂正。未採択の選択肢は判断済みと読まない。
3. 台帳[§9.3.81](../design-notes/technote-ledger.ja.md#mechanism-completion-scope-20261013)（[英語版](../design-notes/technote-ledger.md#mechanism-completion-scope-20261013)）：作者の言葉、A-14の採択、E-4の保存、P-4の診断と当時の未試聴、運用の修理。
4. 台帳[§9.3.82](../design-notes/technote-ledger.ja.md#pitch-movement-requirement-and-steady-20261013)（[英語版](../design-notes/technote-ledger.md#pitch-movement-requirement-and-steady-20261013)）：17:03の試聴の答え、17:07のL5とsteady()の採択、設計と実装の未了。

5. 台帳[§9.3.83](../design-notes/technote-ledger.ja.md#l2-expectation-adoption-and-design-results-20261013)（[英語版](../design-notes/technote-ledger.md#l2-expectation-adoption-and-design-results-20261013)）：17:42のL2の対象・ii-Bの採択、T-18・A-16・A-15の設計結果、統括の試作への判断、L5→L3の接続順。

<a id="l2-expectation-adoption-and-design-results-20261013"></a>

## 2026-10-13：L2の採択と三つの設計からの再開

17:42、作者はL2の対象を反復する拍・分割・小節の位置と支持、T2の入力の役割をii-Bとした。
再来はT2の出来事、位相・強度は既存の速い音響入力から読む。受入は実際の周期と整数の関係で
合い、`metric()`の発音に効くこと。非周期では支持された仮説であり、真の周期と断定しない。
旧T-6のP／Nは未達として保持する。検出の四つの具体値、`steady()`への期待の作用、
間隔側iii、既定変更を採択済みとしない。T-19は本体外の試作で、結果の受領を待つ。

L3はA-15の候補A（保持する行為方針と実効場の増減の転換、作曲者の反応表）、L5はA-16の
案C（既存採用を留まる／調整／移動へ分類）が設計の推奨として返った。統括はA-18／A-17の
本体外の試作へ進めたが、作者の設計採択ではない。新しい連続量0を規則の採択と同一視しない。
二つの試作は繋がず別々に音にする。接続する際はL5の出来事の契約→L3の消費者の順とする。
試作の結果と作者の判断を確認してから本体化を扱い、未採択の案・数値・既定を先取りしない。

設計結果と保存数値の定義は[台帳§9.3.83](../design-notes/technote-ledger.ja.md#l2-expectation-adoption-and-design-results-20261013)
（[英語版](../design-notes/technote-ledger.md#l2-expectation-adoption-and-design-results-20261013)）。
機械に固有の`T-18/report.md`、`A-16/report.md`、`A-15/report.md`も読み、診断・設計・
作者採択・統括の試作指示を区別する。完成の分母は五要件のまま。

以下は過去の判断・結果を確認する際の参照先であり、済んだ単位を再開する指示ではない。

1. [plan-currentの午後〜夜の結果](plan-current.md#temporal-melodic-outcomes-20261010)：止めた線と当時の状態。その後の[2026-10-11の追記](plan-current.md#meter-onset-recurrence-branch-20261011)でT-11の分類とT-13の統合を確認し、[T-16の追記](plan-current.md#metric-basin-initial-rate-20261011)でC＋hardの統合を確認する。§7の20単位とN-7・T-7・H-3の実装・統合記録は保持し、過去の状態を現在の作業指示にしない。
2. 台帳[§9.3.61](../design-notes/technote-ledger.ja.md#metric-phase-separation-stopped-20261010)〜[§9.3.66](../design-notes/technote-ledger.ja.md#pitch-adoption-articulation-prototype-20261010)：位相分離の停止、拍の同期の失敗、動機・phraseの空白、二つの陰性試作、H-1の三素材と読みの訂正、V-1の規則と限界。慣れの既定有効化と実時間は[§9.3.60](../design-notes/technote-ledger.ja.md#habituation-default-on-20261010)、N-7とT-7の取得・受入範囲は[§9.3.58](../design-notes/technote-ledger.ja.md#realtime-acceptance-scope)、文化層と先の試聴は[§9.3.59](../design-notes/technote-ledger.ja.md#temporal-rule-calibration-outcome)。英語版も同じ結果を記す。
3. 機械に固有の追跡外記録 `.orchestration/units/T-11/`の届いた報告とbrief、`beat-sync-check-20261010/README.md`、`participation-check-20261010/README.md`。T-11の原因分類・保存済み36条件、T-12の最終4.06秒の設計、T-13の報告・branch・全テストを読む。台帳[§9.3.67](../design-notes/technote-ledger.ja.md#meter-onset-recurrence-branch-20261011)に採用の状態を記した。T-13は08:28、T-16は08:46にmainへ統合済みである。[設計メモ§4.2](../design-notes/dcc-neurocognitive-hierarchy.md#42-beat-meter-and-groove)でtempo・まとまり・どこで鳴らすかの分担を先に確かめる。
4. 同じローカル記録の`H-1/report.md`の追補と`V-1/registration.md`・`report.md`。V-1の判断材料として用意した三つは`articulation-check-20261010/rearticulated.wav`、`articulation-check-20261010/legato-fast.wav`、`V-1/runs/adopt-s21/audio.wav`。seed 123の変化は`H-1/addition-articulation/runs/fast-s123/audio.wav`と`V-1/runs/adopt-s123/audio.wav`で比較する。fileの共通prefixは`.orchestration/units/`である。
5. `.orchestration/records/decisions-20261007-orchestrator.md`の2026-10-10 13:20以後と`.orchestration/HANDOFF.md`、`.orchestration/status.md`。更新日時だけでなく、時刻付きの追記と各単位の報告を確認する。
6. `AGENTS.md`のProgress and Validation Scope、Multi-agent Orchestration、Author Decision Requests、Git Operation Policyと、`.orchestration/units/COMMON.md`。

追加の入口：[台帳§9.3.75](../design-notes/technote-ledger.ja.md#intrinsic-bodily-rate-20261012)で
身体の固有の速さの判断と範囲を読み、機械に固有の`V-9/report.md`で検査・比較・聴く材料を
確認する。P-1〜P-3の所見、A-9／A-10の根拠は同節にまとめた。相手の時刻に反応しながら
規則的な列を保つ形、二つの実際の速さの維持、拍の層、同期と発音語彙の合成、強弱の弧は
新しい尺度の語だけで解決したとは扱わない。

[台帳§9.3.76](../design-notes/technote-ledger.ja.md#tie-opportunities-20261012)は`tie(p)`の作曲パラメータとしての意味、作者の二本の線の所見、統括の二つの読みの訂正を記す。既定0、配置前の確率、保留とConnect/restの優先を再選定しない。フレーズごとの切替、演奏中の変更、聴こえた区切り、一音ごとの伸ばす／切るは残る。機械に固有の`V-10/report.md`で最終の検査・比較・聴く材料を確認する。

### 2026-10-12の追加の入口：R-2とU-6

実機の実時間は、`2b342bd`、rustc 1.99.0、更新後の依存で四構成×16本を
取り直し、宣言範囲で合格した。B′の初回の外乱ありの不合格と、一度だけの
取り直しは履歴として読む。数値、更新前後の音なし対照、身体込みの余裕と
未計測の範囲、U-6の描画一致・検査は
[台帳§9.3.58の追記](../design-notes/technote-ledger.ja.md#r2-realtime-20261012)にある。
下のT-7の取得は当時の記録として保持し、最新の実機取得はR-2を読む。
13〜16の実時間未確認は今回の範囲で解消したが、別seedの作者の評価は残る。

## 3. 現在の状態と、残るもの

N-7とT-7はmainへ統合・push済みで、H-3の慣れの既定有効化も2026-10-10 22:57に統合・push済みである。D-6着手時点のmainはD-5の文書修正を含む`4da4726`、origin/mainは`39e6840`で、mainが文書1 commit先だった。HEAD・差分、各branchの状態は着手時に確認する。originとの差は`git status`で確かめる。F1は2026-10-11 08:28に四構成の実機合格後、mainの`c2b8308`へ統合済みである。C＋hardを積んだT-16も同日08:46に統括がfast-forwardでmainの`000b884`へ統合した。V-1の本体への実装と語彙は作者が採択済みである。V-2の併用規則は統括が案Aを採択し、branchに実装した。文書のcheckpointへ他のworktreeの作業を混ぜない。校正の記録は`work/t6-calibration-20261008-v1`（`ff9c856`）に残したままで、T-7の本体実装とは区別する。

新しいrendererは楽器でも`render_prototype = true`で選べる。Sine／Harmonicで身体込みの出生8 family、代謝、代表onset footprintが使えるが、Modalは対応外である。旧rendererの再出生の最終選択は方式とfamilyの対応する組だけ身体込みで評価し、新しいrendererではnativeの参照分布がないため身体評価を使わない。

N-7のsourceによる夜の測定は、A′・B′とも宣言範囲で合格し、昼のB′不合格を解消した。A・BはN-7では旧rendererが変わらずWAVが一致するため測り直していなかったが、T-7の最終sourceでは四構成すべてを再測定した。最新の演奏別p99最大はAが3,912 µs（09）、Bが6,769 µs（10）、A′が7,497 µs（09）、B′が9,258 µs（09）で、全48演奏が終了0、出力不足とcallback errorは0だった。BとB′の個々のhopには予算超過があるが、採択したp99と出力不足の基準は満たした。別の日・別の負荷との値の差をmeterの変更の費用とは読まない。取得範囲と数値は台帳[§9.3.58のT-7再測定](../design-notes/technote-ledger.ja.md#t7-meter-remeasurement-20261010)にある。既定は旧rendererのままで、新しいrendererを既定にするか、Modalの対応、nativeの再出生の最終選択の参照が残る。単音holdのPCM一致を波形全体の旧renderer互換へ広げない。

動的代謝の合否は、静的照合（既存上限0.025）、更新の順番と漏れのなさ、同じseedでの再現性で判定する。動的比較は件数と最初の分岐の原因の報告として残す。作者は2026-10-11、B-8の案A0を採り、更新数2を既存の工学の値として[再採択した](../design-notes/technote-ledger.ja.md#body-metabolism-update-count-20261011)。最適値、消費者由来の許容上限、動的な判断の一致の保証とはしない。静的上限を動的scoreの誤差上限にはしない。別seed・別étudeでの発生率や生態の偏りは未検証である。

時間の期待は拍と間隔の二択にせず、既存の作曲者の語彙でどれだけ拍に乗るかを選ぶ。T-6kで整数比の階層の設計を閉じ、T-7で本体への実装と検査を済ませた。初回の確認前は三層の確からしさと比を0に保つ。2026-10-10の作者の指摘で、tempoの好みを文化層の事前分布と整理し、重みを既存の`meter_stability`（既定0）から取る形に直した。内部定数の試聴選択は残っていない。最終sourceの実機計測の入力・取得・集計は`.orchestration/units/T-7/realtime/`に保存し、四構成の合格は台帳[§9.3.58](../design-notes/technote-ledger.ja.md#t7-meter-remeasurement-20261010)に記録した。次の一回型の到来項は既定で無効、検出の数値は未採用のままである。同日の午後、作者は拍の同期と旋律・拍より細かい単位・grooveを分ける判断を採り、metric内で位相を分けるT-8〜T-10を停止した。既存の参加の経路の対照は、約1秒の共通周期を保ち位相が分かれる。拍の同期は07・12と同じpriorでは確認したが、既定のpriorなしとbasin 2.8〜3.4で失敗が残る。T-11は二つとも設計判断と分類し、修理せずに停止した。F1の続きはT-12最終設計CをT-13へ実装し、作者の採択と2026-10-11の実機の四構成合格後、mainへ統合済みである。F2は作者がC＋hardの最小形を採り、T-16で実装し、同日08:46に統括がmainへ統合した（台帳[§9.3.61](../design-notes/technote-ledger.ja.md#metric-phase-separation-stopped-20261010)・[§9.3.62](../design-notes/technote-ledger.ja.md#metric-beat-sync-regressions-20261010)）。

動機・phraseの2〜10秒の音高・時刻の共同の生成状態は空白である。P-A1とP-B1は、既存の判断へ提案だけを渡す条件で陰性の結果を返した。学んだ意図に重みを持たせる続きは未着手・未採択で、優先を下げた。既存の語彙と慣れのlegato-fastは作者が「かなり自然なメロディー」と評価し、聴かせないanchor、一体の近似自己除外、遅い／速い二体も作曲の材料として評価した。これは一般的な系列生成の受入ではない。慣れはH-3で既定有効化済みで、scriptから切り替える語彙は無い。

V-1の外部試作は、新目標の採用一回に発音一回、移動の始まりで新音高をattackする規則を確認した。作者は2026-10-11に試聴後、本体への実装を採択した。同日09:03頃に `pitch_apply_mode("rearticulate")` の形と名前も確定した。決定済みの二点と語彙を再選定せず、V-2は既定を保ってHoldの自由音高のVoiceへ実装し、glideとのlive切り替えを検査した。統括が技術的な併用規則として案Aを採り、自由音高のGatedとの併用を最終specでerrorにする。liveの対象外の指定も状態へ書き込む前に拒否する。12 étudeのWAVと意味のあるreportはmainと一致した。新方策の聴く材料とofflineの費用の参考値はV-2の報告に置く。

作者が求める最終形は、フレーズによってrearticulate、legato、同音高での打ち直しを混ぜ、Voiceが一音ごとに選ぶ形である。Populationごとの固定modeとscriptからの切替えは途中の形であり、A-1（T-5b）が一音ごとの選択を別に設計する。V-2は一音ごとの選択を実装せず、採用時の方策判断を一か所にまとめ、採用の有無に依存しない再発音経路の境界を検査した。公開の同音高の発音要求も未実装である。要求と本体実装を区別する。[台帳の2026-10-11追記](../design-notes/technote-ledger.ja.md#pitch-rearticulate-native-decision-20261011)と[計画の同日追記](plan-current.md#pitch-rearticulate-native-decision-20261011)を入口にする。

2026-10-11の次の判断では、作者が「スタッカート自体は音楽表現としてありえる。それが意図的ならば」と述べ、A-2の持続＋参加を本体へ入れる方向を採った。V-3はsustainとentrained／flowを両方明示した時だけこの形へ下ろす実装・検査・文書をbranchに用意した。書く順には依存しない。待つ・見送る時も音を保ち、時計の発音はその時点の身体音高で打ち直す。採用時の候補は既存の費用で競合させ、採られなければ滑る。保存したA-2場面で採用時の発音が0件だった限界を保つ。自由音高への明示のgate_snapはerror、省略時はGlideとする統括の技術的な規則を実装した。V-3には、採用済みの目標へ機会の発音で跳ぶgate_snapを含めていない。作者は2026-10-11、`sustain()`と`entrained()`／`flow()`を両方明示した時だけ持続＋参加にする語彙を確定した。統括が同日、実装と文書をmainの`23a550b`へ統合した。[台帳§9.3.69](../design-notes/technote-ledger.ja.md#sustained-participation-native-20261011)と[計画の追記](plan-current.md#sustained-participation-native-20261011)を入口にする。

- **rearticulateの境界（実装済み）**：作者は2026-10-11、作曲者が境界を置く`rearticulate_above(cents)`、省略時0を採った。V-4で最後の実発音からの正味の変位を測る境界を実装し、検査と日英文書をそろえた。[台帳§9.3.71](../design-notes/technote-ledger.ja.md#articulation-boundary-and-deferred-target-20261011)を参照する。
- **持続＋参加のgate_snap（実装済み）**：作者は2026-10-11、採用した行き先を保留し、Voiceが鳴らすと決めた機会の発音で跳ぶ意味を採った。V-4で実装・検査・文書をそろえ、省略Glideを保持した。[台帳§9.3.71](../design-notes/technote-ledger.ja.md#articulation-boundary-and-deferred-target-20261011)を参照する。
- **音を切って間を置くrest（実装済み）**：作者は2026-10-11、候補A、全Toneの終了を起点にするG1、持続＋参加への限定、名前`rest`を採った。V-5で有限音長を独立に保持し、`rest()`は固有周期一つ、`rest(periods)`は作曲者の倍数とする形を実装した。全音を有限長で切り、一音ごとに伸ばすか切るかを選ぶ候補Bは未着手。検査・取得の範囲と設計メモのGapとの違いは[台帳§9.3.72](../design-notes/technote-ledger.ja.md#finite-notes-and-rest-20261011)に記した。
- **持続＋参加の境界つきRearticulate（実装済み）**：作者は2026-10-11にA-6の合成、交点cancel、明示の持続＋参加＋`pitch_apply_mode("rearticulate")`を採った。小さい採用は旧保留を取り消して滑り、大きい採用は鳴らす機会まで保留する。V-6でrest、演奏中の三mode／境界変更、出生時の空状態と既定対照を検査した。フレーズ単位は作曲者とVoiceの両方という要求を保ち、この段階ではVoice側の選択は設計対象として残った。[台帳§9.3.73](../design-notes/technote-ledger.ja.md#bounded-sustained-participation-20261011)を参照する。
- **発音方針の保持（実装済み）**：作者は2026-10-11にA-8の形を採った。V-7は`articulation_persistence(p)`を配置前の初期条件、既定0、自由音高の持続＋entrained／flow＋Rearticulate＋正の境界に実装した。templateのpだけを継ぎ、再生成はBaseから始まる。mode変更はBaseへ戻し、同modeと境界だけの変更は保持を続ける。pはつなぐ割合ではなく、Baseにも働く惰性で、秒・拍・聴こえたフレーズ境界とは結び付かない。統括の「形は一音ごとに入り交じる」という言い過ぎ、固定Baseで見つけた停止後の接続の誤りと修理、検査・取得の範囲、一音ごとの伸ばす／切る・境界推定・学んだ意図の重みという残りは[台帳§9.3.74](../design-notes/technote-ledger.ja.md#articulation-policy-persistence-20261011)に記した。
- **創始の身体の固有の速さ（実装済み）**：作者は2026-10-12、entrained／flowの身体の初期条件に`intrinsic_rate(hz)`を置く判断を採った。共通の固定格子は戻さず、固有値を生存中保ち、参加周期だけが既存の適応で動く。音長・rest・重なり予測も同じ周期を読む。省略は従来の2 Hz、metricのbasinとpulseは対象外、再生成はtemplateから始める。有限0.01〜200 Hz以外は対象内の配置前errorとし、対象外は警告して無視する。二本の線の耳の条件、統括の三つの説明の訂正、A-9／A-10の根拠とまだ足りない表現は[台帳§9.3.75](../design-notes/technote-ledger.ja.md#intrinsic-bodily-rate-20261012)に記録する。

2026-10-11 12:50頃、作者は発音の境界を作曲パラメータとし、`rearticulate_above(cents)`を採った。最後の実発音からの正味の変位で選び、省略0は従来の採用ごとの発音を保つ。正の値を既定へ置かない。統括が固定の規則として入れようとした誤り、A-3とA-3bの結果は台帳に残す。作者は持続＋参加の明示のGateSnapも、A-2cの一つの目標を機会まで保留する意味へ替える判断を採った。V-4は両方を実装し、Glideとの切り替え、取消、継承、既定の描画対照、身体と両renderer、聴く材料と日英文書をそろえた。2026-10-11にmainへ統合した。検査結果と未確認の範囲はV-4の最終報告と[台帳§9.3.71](../design-notes/technote-ledger.ja.md#articulation-boundary-and-deferred-target-20261011)、[計画の追記](plan-current.md#articulation-boundary-and-deferred-target-20261011)に残す。V-5は有限音長と`rest`を実装し、GateSnapの保留を発音中・余韻待ち・間のあいだ保持する。任意負荷の実時間や一音ごとの切断選択を実装済みと数えない。

聴こえる音量と生態入力の分離は保留であり、身体ampやADSRの変更を独立した出力gainの代用にしない。この未決を先回りして実装へ進めない。

揺らぎの語彙も未決である。H／RとT2の聴覚前処理の共有は計画しており、作者の2026-10-09の指示は実施時期の延期であって取り下げではない。これらは[plan-current.md §7](plan-current.md#7-結果)で追跡する。

## 4. 2026-10-07の開始手順の履歴

旧版は、未コミットの計画改訂をworktreeで固定してから、段0・段1のU-1〜U-4、T-1／T-2、B-1／B-2、M-1を割り当てる手順を示していた。その土台作業と後続の単位は、§2で参照した結果に置き換わった。旧版の開始時HEAD、bufferの残存状態、最初のwave案を、次の作業指示として使わない。2026-10-07の採択原文と根拠は`.orchestration/records/author-instruction-review-20261007-v1/`の`adopted.md`と`proposal.md`に残る。

## 5. 注意

- commitは共有checkoutで行わず、指定worktreeのbranchで行う。commit前のClippyとpre-commit hookを省略しない。hookは `cargo fmt --all` と `git add -u` を実行するため、追跡差分が意図した対象だけであることを先に確認する。`--no-verify` や `core.hooksPath` の上書きで迂回しない。
- gitに無い内容を持つworktreeは削除しない。2026-10-07にindexが全削除だった `f2-c2-*`、`f2-nominal-resource-prototype-20261007-v1`、`f2-shared-control-feature-prototype-20261007-v1` は、内容がディスク上にしか無かった。整理するときは現物を確認し、保全と作者の指示を優先する。
- 統括の記録は `.orchestration/` に置く（`status.md`、`units/<単位>/` のbriefと報告、`records/`、計時のlock）。`target/` には唯一の写しを置かない。2026-10-07 に共有checkoutで `cargo clean` が実行され、`target/` 以下の追跡外の記録が失われた。復元できたのは `adopted.md` と `proposal.md` だけである。
- workerの起動直後はsessionが未確立である。`agent-shell--state` の `(:session :id)` が入ってから `conchordal-worker-send` し、送信後は `~/.codex-keio/sessions/` のrolloutかbufferの伸びでturnの開始を確かめる。workerのbufferは `conchordal-worker-frame` で別frameに並べる。
- 計算前に `.orchestration/units/COMMON.md` の「メモリ」を読む。MemAvailableが12 GiB以上あることを確認し、所有jobだけへ8 GiBの上限を掛ける。user scopeが使える場合は `MemoryMax=8G`・`MemorySwapMax=0`、sandboxで使えない場合は `RLIMIT_AS` 等の代替を使う。端末やEmacsのscope、slice、`MemoryHigh` は変更しない。cargoの並列は8以下とし、PID・実際のlimit・exit・可能ならRSS peak・killの出所を記録する。
- `CARGO_TARGET_DIR` は系統ごとに一つを使い回す。単位ごとに新しいworktreeを作ってcacheを複製することはしない。
- 計算前は共有checkoutの `/home/shafi/lwrk/conchordal/scripts/timing_lock.sh check` で排他枠を確認する。worktreeの古いscriptは使わない。合否に使う計時は、1セルの試行の後に排他の時間枠で行う。
- R-1の実機測定は統括のshellから行った。workerのsandboxは音声deviceとuser busに届かず、`--play true` が初期化で止まりprocessが残った。取得・集計のscriptは `.orchestration/units/R-1/realtime-20261008/run_r1.py` と `analyze_r1.py` にある。
- 排他の時間枠は別projectのjobに効かない。R-1のBのv2では、別projectの検査jobによって全処理が約3倍遅くなり不合格だった。測定の前後にload averageと `journalctl --user` のjob開始・CPU消費を確認し、外乱ありの不合格と静かな状態での再測定を別々に保存する。
- `agent-emacsclient --eval` から確認dialogを出しうる `find-file-noselect` や `revert-buffer` を呼ばない。2026-10-08にGTK dialogがEmacs全体を止めた。呼出しには `timeout` を付ける。
- 作者へ判断を求めるときは、推奨ごとに、採ると発生する作業、採らないと消える作業、数値の由来、後から変える費用を書く。
- 作者の未決判断はworkerが決めない。pushと、gitに無い内容を失う削除は明示の指示を待つ。

<a id="habituation-default-branch-20261010"></a>

### 2026-10-10：慣れの既定有効化をmainへ統合し、push完了

作者は慣れ（`[psychoacoustics.habituation]`）を既定で有効にする判断を採った。
`main`の`1f55a45`を基準に、`work/h3-habituation-default-20261010-v1`へ実装と検査
（`6a7b6b2`）、文書同期を別commit `39e6840`で用意した。
同日、作者の指示を受けた統括がmainへfast-forwardで統合し、22:57にpushした
（`1f55a45..39e6840`）。push後のmainとorigin/mainはともに`39e6840`だった。
既存の5秒・8秒・0.25と慣れの式、上位の12 étude、Rhaiの語彙は変えていない。
科学的な対照とbit一致の検査には明示offを使い、凍結済みの登録値・hash・結果を保持した。

慣れを有効にした四構成は、統括の実機取得で既に合格している。演奏別p99の最大は
Aが5,896 µs、Bが6,196 µs、A′が7,442 µs、B′が9,308 µsで、採る全48演奏が終了0、
出力不足とcallback errorは0だった。初回B′の不合格、B・A′のメモリ入口での中断、
取り直しの窓とA′の外乱を含む経過は、台帳[§9.3.60](../design-notes/technote-ledger.ja.md#habituation-default-on-20261010)に記録した。
実時間は取得し直していない。branchでの全テストと既定設定による12 étudeの全長offline描画の記録は
`.orchestration/units/H-3/report.md`にある。20単位の分母を増やさず、作者の試聴評価と実時間の合格を分ける。


<a id="meter-onset-recurrence-branch-20261011"></a>

### 2026-10-11：T-13のbranchを読む入口

**08:28の追記。** T-13の実機の四構成は全48演奏で合格し、mainへ`c2b8308`として統合済みである。以下はbranchを渡した時点の入口を保存した記録であり、採用・実機の計測を待つ現在の指示ではない。今の続きは末尾のT-16の入口を使う。

mainの基準は`217c7e8`。実装先は`.worktrees/t13-meter-recurrence-20261011-v1`、branchは`work/t13-meter-recurrence-20261011-v1`である。実装・検査`1000b99`と文書の二commitを用意した。mainは動かさず、pushしていない。次は`.orchestration/units/T-13/report.md`の07の端点の敏感さ、雨、仕事量と検査の限界を読み、F1をmainへ採るか作者が判断する。実機の計測は未実施であり、統括が作者に知らせてから四構成の取得を行う。過去のT-7・H-3の合格をこの新しい閉ループの合格へ読み替えない。F2とV-1の選択は独立して残る。

### 2026-10-11：作者判断後のT-16を読む入口

現在の判断は[計画の追記](plan-current.md#metric-basin-initial-rate-20261011)と[台帳§9.3.68](../design-notes/technote-ledger.ja.md#metric-basin-initial-rate-20261011)。上のT-13の採用待ち・F2未決は当時の記録である。採択済みの形はC＋hardであり、Cはmetricだけ、再生成はPopulationの初期設定、basinはsceneの最後の一値、明示Pulseは対象外とする。

T-16着手時のmainは`217c7e8`、統括のF1統合後は`c2b8308`だった。T-16はそのT-13の最終先端の上の`work/t16-basin-voice-rate-20261011-v1`へ実装し、2026-10-11 08:46に統括がmainの`000b884`へfast-forwardで統合した。機械に固有の`.orchestration/units/T-16/report.md`にcommit、検査、12 étudeの照合を記す。timing lockが保持されている間はcargo・test・renderを始めない。最終sourceの全テスト1,044件と二つの18条件のassay、F3は通過した。既定の12 étudeはT-13とWAV・意味のreportが一致したため、実時間は取得済みのT-13の四構成で足りる。実機の計測は取り直していない。共有checkoutのテスト記録はT-16の最終記録へ置き換えた。branchの完了、mainへの統合、push、作者の試聴は区別する。


<a id="polyphonic-samples-author-selection-20261012"></a>

### 2026-10-12：採った四本のサンプルと残る不足

作者はE-2の四本を以前のsamplesより音楽的と評し、E-3との聴き比べでも四本ともE-2を選んだ。
Three Windowsでは「動きがすくなくなってpolyphony感がへってる」、ほかも程度の差はあるが同様、
という答えである。E-3の境界・身体速度・参加への変換・tieの改稿は採らず、E-2の順で
Three Windows、Stone and Wire、Folded Air、Open Courtを13〜16に置いた。先頭comment以外の
codeと既存12本を保ち、サンプル集を16本とした。intrinsic_rate・tie・rearticulate_aboveは
既定の挙動を変えず、作曲者が選べる語として残す。

朝の正常性の問いを消す依頼と読んだ誤り、Three Windowsの複数の変更を一つに重ねて
原因を切り分けられなかった誤りは、保存記録とともに[台帳§9.3.77](../design-notes/technote-ledger.ja.md#polyphonic-samples-author-selection-20261012)へ記した。
発音数だけで作者の「動き」を測らず、抜けと格子からのずれを分ける。13〜16の実機の実時間、
特に9体のOpen Courtと、別seedの作者の試聴は未確認である。同期と発音方策の同一Voice合成、
強い時刻応答と規則的な列、独立した強弱の弧は残る。機械に固有のS-1報告に検査と保存音の照合を置く。


### 2026-10-12：samples全体の見直しの後の二群

作者は、1〜12を機能の実演、13〜16をそれらを組み合わせた曲として、同じsamples/内で
書き分ける判断と、音を変えない小さい修正を採った。READMEと本の道筋は、共有拍への結合、
音を聴く参加、拍を作らない流れという現行の三つの方策へ直した。07のepigraphは専用の
拍Voiceを持つ実態へ合わせ、13〜16は語・数値・順序を保って整形した。

08の扱いは作者の試聴後に決め、fileと一覧行を保つ。産出側の拍の読みは一seedの所見で、
耳の評価ではない。7〜10・12の細かい定数、指定音高から自由に動く初期配置の語彙の穴、
発音の語彙の短い実演、13〜16の別seedの評価と実機の実時間は残る。
[台帳§9.3.77](../design-notes/technote-ledger.ja.md#polyphonic-samples-author-selection-20261012)の追記と、
機械に固有のS-1/brief-2.md・report.mdに、見直しの所見、判断、保存音の照合を記録した。

### 2026-10-12：08 Murmurationの説明と作者の判断（21:20頃の更新）

統括の「08は主張が成り立っていない」という所見は、機構の指標から聴こえ方を
断定した誤りとして撤回する。作者が08を徐々に歩調が合うように聴いた印象が、
撤回のきっかけとなった。

作者は08をentrained()のsampleとして残し、「entrainとその実装から自動的にいえる
ことをメインで、私がさっき感じた主観はいれないで」と指示した。直前の
「file・一覧行・説明を変えない」は統括の読みで、この判断に置き換わる。
08のcodeは保ち、comment三か所とsamples/README・日英bookの8の行を、自身の
間合いを基準に聴こえた再来へ周期を寄せ、共有拍の位相を狙わない実装の意味へ直した。
作者の印象はsampleの説明の根拠にしていない。結果・訂正・実装の出典は
[台帳§9.3.77](../design-notes/technote-ledger.ja.md#polyphonic-samples-author-selection-20261012)と
機械に固有のD-11/brief.md・brief-2.mdにある。

今回、beta-evaluation.mdの08を「同期の立ち上がり」の評価対象とする行は直していない。
時間の聴き手のモデルの開いた問いとして、同じ取得で作者の印象と拍の確からしさの読みが一致しなかった原因は未確認であり、数から決めず、新しい単位・検査は起こさない。


### 2026-10-12：atと明示の移動

作者が22:57頃に採ったA-13のC1により、seek_consonanceの明示の意図とFreeが
配置時に残るVoiceでは、at(f)は初期位置だけを指定する。既定のVoiceや、後から
freq／anchorで固定へ戻したVoiceは従来の固定atとなる。freq→seekとanchor→seekは
初期位置、seek→freqとseek→anchorは固定音高である。新しい配置語は無い。

初期配置の語彙の穴を未処理として再開しない。V-11の保存音の照合とgroundの扱い、
検査・未確認の範囲は[台帳§9.3.78](../design-notes/technote-ledger.ja.md#at-initial-explicit-movement-20261012)と
機械に固有のV-11報告を読む。作者の新しい試聴や実機の実時間の受入とは分ける。


### 2026-10-13：持続と同期の併用

作者の「推奨」を受け、統括が09:28頃に明示した推奨に従い、sustain＋metricを
持続＋同期として実装する。
2026-10-13の09:31頃（transcriptの時刻は09:31:13 JST）、作者は統括が伝えた読みに対して「入れる。」と明示で答え、この読みを確認した。
初回Holdと拍の整数交差を既存の発音resolverへ接続し、
採用の無い拍も打ち直す。伸ばす指定は自由音高のGateSnap／Rearticulateのtieを使う。
共有の拍は音から確認され、未確認の間は既存の速度・位相から自走する。外から拍を置く語ではない。
固定音高は持続＋参加と同じ扱い。暗黙のCoupledと単独のプリセットは保つ。

V-10のconst genericは実行時のOption判断へ整理する。実装・検査と文書の範囲、
A-12の対照、作者の答えと統括の読み、残る表現は
[台帳§9.3.79](../design-notes/technote-ledger.ja.md#sustained-synchronization-20261013)へ記録する。
機械に固有のV-12報告で最終の検査と既定・主場面・試作の保存出力の照合を読む。
これを新しい聴感評価や実時間受入としない。旧20単位の分母と次の単位は増やさない。

### 2026-10-13：持続＋同期の省略tieは伸ばす

作者はE-4を聴き、全拍の発音が不自然になる場合と、6の採用拍だけの形への所見を述べた。
14:36に推奨を採り、自由音高のsustain＋metric＋GateSnap／Rearticulateは、tieの省略を1と同じにする。
採用先を保留した拍で打ち、ほかの拍では伸ばす。毎拍打つにはtie(0)。明示のpは保ち、
持続＋entrained／flowの省略値0、固定音高、Glideは変えない。live modeの適用中と再出生も同じ規則。

全拍を既定にした機構とE-4のbriefの経緯、統括が発音数と動きを取り違えた二つの落ち、
作者の言葉、実装と照合の範囲は[台帳§9.3.80](../design-notes/technote-ledger.ja.md#synchronized-tie-default-20261013)に記録する。
§9.3.79と先の併用記録はV-12時点の経過として残し、現在の省略値はこの追記を使う。
機械に固有のV-13報告で正規検査、41件の不変と15件の意味対応、pinを読む。
新しい聴く材料や実機計測、次の単位は起こさない。「上の音」の同定と音量、作者の新しい試聴は未確認。
旧20単位の分母は増やさない。
