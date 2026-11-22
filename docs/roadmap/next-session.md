# 次のセッションの始め方

2026-11-22更新（簡素化のS0）。作者が認めた構成の統合（DR-1）は `3180847b` でmainに入った。2026-11-21からは[簡素化の指示書](simplification-directive-2026-11-21.md)の期間（2026-12-25まで）である。mainへ新しい原理や機構を入れず、研究と不採用分はローカルの研究branchへ退避commitする。
経過は、2026-11-19の設計の見直しと作者の二つの採択（認められた構成をまとめる、慣れの既定を鳴った所の順応に変える）に、11-21の簡素化（S1〜S4）と11-22のS0を加えた再開点である。
曲の段の目標はbetaデモ。新しいセッションで、古い「自己除外の比較試聴待ち」「残るのはv0.5.0公開だけ」「方向Bのデモを進める」へ戻さない。

## 最初に読むもの

1. `AGENTS.md` と[現行計画](plan-current.md#current-stage)。承認された範囲、既定値、作者判断と科学的受入の区別を確認する。
2. 台帳[§9.3.123](../design-notes/technote-ledger.ja.md#harmony-decisions-20261103)・[§9.3.124](../design-notes/technote-ledger.ja.md#harmony-action-consequences-20261103)。A1/B1/C1は採択済み、C1は実装済み・作者試聴保留という順序を読む。
3. 台帳[§9.3.117〜118](../design-notes/technote-ledger.ja.md#incremental-endless-scores-20261031)。endlessの実装と、IP-ACQの受入を読む。V5未取得、V6合格、V7は2×10⁷発行まで合格、V8の差分推定と実機再取得を混ぜない。
4. 台帳[§9.3.126](../design-notes/technote-ledger.ja.md#voice-self-exposure-one-listener-20261104)・[§9.3.127](../design-notes/technote-ledger.ja.md#voice-melody-stop-20261105)と[現行計画のVoiceの節](plan-current.md)。VB-3は統合済み、TU-2／TU-3は統合しない。§9.3.127の停止は、旧来の経路と文脈による選択の試作についての記録である。デモは方向Bで進めず、11-07の基準を保って待つ。
   台帳[§9.3.129](../design-notes/technote-ledger.ja.md#art-form-path-20261106)で、芸術の形態としての経路、ライブのみで録音を公開しないこと、browserの版を既定の配布の候補とすること、共有世界とAir-Gapの範囲を読む。
5. 台帳[§9.3.132](../design-notes/technote-ledger.ja.md#first-impression-cue-response-20261107)〜[§9.3.137](../design-notes/technote-ledger.ja.md#requirement3-settling-closed-rc-20261113)と、[現行計画の「第一印象の基準と、動く理由の問題」](plan-current.md#first-impression-and-gestures)。
   四つの要求、送り手の音型（GT）の結果、和声進行の定義と評価（HP-1／HP-2）、要求3のA'を閉じた経緯、途中で分かった素材の性質を読む。
   GTの計画の正本は `.orchestration/units/GT-RF1/reframing.md`、考え直しの材料は `.orchestration/units/ROOT-1/materials.md`、A'を閉じた記録は `.orchestration/units/RC-1/place-closure.md` と `time-closure.md`。
   GTで届いたのは主張2の限定した支持と、作曲者が与えた入れ替わりの下での3aまでである。変異の分布と選択の向きには解き方がまだ無い。
6. 台帳[§9.3.138](../design-notes/technote-ledger.ja.md#flora-respond-to-onsets-rs1-20261114)〜[§9.3.142](../design-notes/technote-ledger.ja.md#design-review-consolidation-20261119)と、[現行計画の「線の疲れと慣れの形、感じられた拍、設計の見直し」](plan-current.md#consolidation-20261119)。
   苔の応答、雨の語、慣れの既定の変更、感じられた拍の一段目と閉じた二段目、設計の見直しと統合の範囲を読む。
   慣れの既定は `drive = "channel_p"` である。デモの曲は、作者が 2026-11-20 に採った Line（旧「線と苔（鋭い応答）」）と Rain（旧「雨と苔」、雨はポツポツから入る）の二曲で、どちらも既定で鳴らす（残りの四曲は目録から外した。台本は `3180847b` にある）。「線と苔（鋭い応答）」は、DR-2で線の温度0と移動の費用0.125の版に替え、作者が聴いて残した。DR-3の線の音の版（既定のbrain、`sustain_drive(1.0)`）は、作者が「悪化した」と答えたので戻した。線の倍音とピアノ・擦弦の比較と、既存の語だけの対照（版U。作者は「かわらない」と答えた。曲の中では線の倍音が苔に埋もれていた）は `.orchestration/units/DR-4/` にある。線を苔より前に出した版（`.orchestration/units/DR-5/`）にも作者は「全く変わってない」と答え、指している音が違うのではないかと指摘した。線・苔・地衣に分けた音（`.orchestration/units/DR-6/`）で、作者は「線のことを言っている。音がきたない、荒削り」と答えた。原因の切り分け（`.orchestration/units/DR-7/`）で、荒さは倍音の側にあると分かった（倍音なしだけが違って聴こえ、それはさみしい）。倍音の作りを選ぶ五本（`.orchestration/units/DR-8/`）では、倍音を弱くした作りがいちばんましだが、きれいではないと答えた。響き・音高の揺れ・重ねを一つずつ足した音（`.orchestration/units/DR-9/`）に、作者は「やるとしてもほんのすこしがよさそう」と答え、Conchordalへの入れ方（特に響き）を問うた。響きは一つに決めず、一旦保留（音の道筋の段：体の中、体と体の間 = GitHub issue #2、場所、聴き手の空間 = issue #3）。重ねを先に進め、体の写しの作りを直した（`spread` が重ねの量）。ほんの少し重ねた版（`.orchestration/units/DR-10/`）は「聞こえない」だった。聞こえ始める量は s = 1/3（作者「4」）。その量の台本の版 B0Q と、ほかの曲に弱めに重ねた版（DR-12）は、作者が聴いて5曲の台本に入れた。設計案の第2版（Astraのレビュー反映）は `.orchestration/units/DR-13/design-v2.md`。作者の響きの設計案とその独立レビュー、統合した設計案は `.orchestration/units/DR-14/`。響きの第1段階は実装と主比較まで終えた（`.orchestration/units/DR-16/`〜`DR-23/`、当時の worktree `room-stage1-20261119`。`dr1-consolidate-20261119`の未commitの変更をすべて含む）。作者は出力の部屋の盲検の試聴（`.orchestration/units/DR-22/`）で響きの量を「ちょうどよい」と答えた（2026-11-20、音はモノラル）。出力の部屋は研究の選択肢として残す。作者はステレオ化を採り、DR-24 で実装した（`.orchestration/units/DR-24/`）。部屋が on の時の既定の出力 gain は DR-25 で −10.5 dB にした（off は +4.5 のまま。乾いた較正は今の版で +4.0 dB 相当なので、統合後に再較正した）。W/W は `12_emergence_and_resolution` で予約が 27,181 本・約 77 GB になり描けない。実際の同時は 19 本（DR-27）だが上界の証明ではないので、採る時は読者の和集合と退役期間から静的な上界を求める（DR-28）。作者の指摘から試した予測による自己抑制（DR-26/27）は No-Go（誤差ゼロでも判断が変わる）、自分の余韻を聴いても線は間を取らなかった。Astra への相談（DR-28）で、別の答え（乾いた側で自分を正確に引き、外の音だけを軽い部屋に通す）が出た。演奏の文献ではテンポの向きは一様でないので、大聖堂で遅くする線は閉じた。作者の試聴（DR-29、D/W と W/W の盲検の二組）は組で割れたので No-Go。W/W は研究用のまま凍結し、軽い部屋 q も作らない。第 1 段階は出力の部屋（D/W）だけで閉じる。

   room-stage1 は `dee5a91d` で main に統合済み。第 2 段階からは揺らぎの修理、打撃過渡、契約草案 A2 の改訂が main への採用分である（DR-46）。不採用分と身体の遺伝の研究は `work/body-phase2-20261121`（`bbeb9bf9`）、単位 E（DR-40）は `work/room-stage1-20261119`（`6baabdf4`）に保存した。両方ともローカルの研究 branch に留める。基点にある demo-20261116 の作業ツリーの写しは、2026-11-20 の時点でそのセッションの作業ツリーと全ファイル一致した（そのセッションは 11-17 から動いていない）。写しと felt-beat 一段目（作者が研究の設定として採った）は main に統合済み。
   見直しの記録は `.orchestration/units/DR-1/review.md`。旋律の生成と、拍より上のまとまり（二段目）は開いていない。
7. 台帳[§9.3.125](../design-notes/technote-ledger.ja.md#listener-prediction-e0)と[Listener Twin Design](../design-notes/listener-twin.md#gui-1-presentation-prediction-and-current-display-contract)。E0は未較正の混合音響baselineであり、DES-1／DES-2全体の実装ではない。後続のGUIの到達点は `41133692`。
8. [統括とhostの運用](../operations/orchestration.md)。単位記録の所在、既存job、timing lock、再取得の制約を確認する。必要な設計は[文書の入口](../README.md)・[設計ノート索引](../design-notes/README.md)から探す。

## 進行中の長い取得を先に確認する

**ここに記す稼働状態は統括記録に基づく。新セッションでは実際のserviceと最新出力を確認する。**
DOC-3からはuser busへ接続できず、実際の稼働・終了・freeze状態は確認できていない。
既存取得の再起動や、別のsourceでの取り直しを始める前に、固定sourceとbinary、既存出力、真の終了状態を確かめる。

- **統合（DR-1）とworktreeの整理（S0）。** 統合は `3180847b` でmainに入り、demo-20261116の写し（`0bc8db99`）とfelt-beat一段目（`8255cf21`）も同じ時にmainへ入った。
  - 2026-11-22のS0で、共有checkoutのmainを `origin/main` に揃え、worktreeを整理した。未commitの変更は、worktreeが乗っていたbranchの先端へ、detachedのものは `archive/wt-<名前>` へ、未検証の退避commitとして残した。dr1-consolidateとdemo-20261116の未commitは、照合でmainに吸収済みと確かめた上で、同じく退避commitにした。
  - worktreeの中にあった取得の記録、`test_report.txt`／`test_status.txt`、worktree内の `.orchestration/` は、`.orchestration/records/worktree-leftovers-20261122/<名前>/` へ移した。消したのはcargoのビルド出力だけである。各worktreeの処置は `.orchestration/units/SIMP-S0/run.tsv` にある。
  - 統合に入れていないもの：GTとGT-E（`research/gt-gesture-20261109`）、苔の聴く範囲の語（`research/rs1-respond-20261114` の退避commit）、`habituated_growth`（`.orchestration/units/MO-4/` のpatch）。
- **IP-ACQ V7：推定器の精度。** 最初の取得で2×10⁷発行（演奏約55時間ぶん）まで全件合格。3×10⁷の取り直しは作者の指示で止めた（変わる判断を名指せない）。
  取得も再開もしない。55時間を超える連続演奏が必要になった時に、その必要を消費者として取り直しを判断する。
- **IP-ACQのV6・V8は取得済み。** V6は譜面132,000秒を合格、V8は計測交絡の限定切り分けと疑い2条件の実機再取得まで記録済み。
  V6/V8を未取得へ戻さない。
- **デモへの道の再整理（RP-1）。** 2026-11-13にFableの助言が届き（`.orchestration/units/RP-1/advice.md`）、作者は要求3の欠け（A'とB）を最初の作品の条件から外すことを、仮のゴールとして採った（`RP-1/decisions.md`）。作者は入れ替わりを使うこと（作曲者が与えた札付き）と、統括の叩き台（`.orchestration/units/WK-1/promise-draft.md`）をそのまま宣言した。台本 `samples/research/first_work.rhai` を描き、記録の門を通して作者が一回聴いた。答えは「いいえ、要求1」（雨は協和でもtemporal DCCでもなく、Floraは無調。`.orchestration/units/WK-1/listen-result.md`）。作者は追加で「7分間ずっと変化がないので退屈」と答えた。
  - 2026-11-14、作者はRP-2の推奨を採った（`.orchestration/units/RP-2/decisions.md`）。分の尺度の変化は作曲者の形式（部品の出入りと区間ごとの条件の変更）で作り、GT-E（GTの写像で場の峰と拍の位置の取り分を広げる）を開く。雨は作品から外し、別個に残す。
  - GT-E は研究branchのworktreeに実装した（未commit、`.orchestration/units/GT-13/`）。音高の側は門を通った。時刻の側は発音の87〜90%が拍の位置に乗り、二つの門に当たった。
  - 作者の依頼で六つのデモを描き（`.orchestration/units/WK-3/`）、作者が聴いた。答えは「temporal DCCを感じられるものはひとつもない」。線と苔（P1）は単調さを除けばアンビエント音楽として成り立つ。群れや拍に乗る声を足すと苔と別個に聴こえ、一緒にいる必然性が無い。線と雨では線が邪魔（`WK-3/listen-result.md`）。
  - 作り直しの相談（RP-3）の推奨を作者が採った（拍の位置合わせはこの作品では止める、群れは入れない、GT-Eは研究branchへcommit `d30d3f9c`）。部品のあいだの応答（RS-1）を開いた。
  - 苔の側：Floraの `respond_to_onsets`（外から聴こえた立ち上がりの後に生態の時間を速める、既定0）。作者は30秒の対で「応じている」と聴き、P1の苔に強さ8を足した版の全長で「temporal DCCを感じる。現時点の一つのデモとしていい」と答えた（2026-11-14、`.orchestration/units/RS-9/decisions.md`）。台本は研究branch `research/rs1-respond-20261114` の `samples/research/demo_p1r_w8_s20261109.rhai` で、2026-11-22のS0で未検証の退避commit `3c353530` に入れた。
  - 線の側（Voiceの `respond_to_onsets`）は門を通らず、作者が閉じた（既定0のまま研究branchに残し、mainへ入れない。RS-7、RS-8）。
  - 次は、保存（commit）の判断、作品の値の宣言、mainへの統合、ページの公開の扱いを作者に問う。A'とBは本来のゴールとして残る。
- **雨（RW-1〜RW-12）。** 作者の求め（激しい時にこそ場、弱い雨から、強弱は1/fだけ、協和した所ほどよく響く）に答える雨を作った。研究branch `research/rain-ring-20261114`（未commitだった分は2026-11-22のS0で未検証の退避commit `54c95500` に入れた）。雨が自分の滴を除いた場を読む `leave_drops_out`、鳴り終えた共鳴体が滴の高さへ移る `objects_follow_drops`、協和な所の滴ほど長く鳴る `ring`、滴の高さのずれを場の解像度に収める `width(0.0052)`、滴と共鳴体を整数比の倍音にする素材（`harmonic_modes()`）。作者は場を読まない対照との対で「違いはきこえる」と答えた（RW-12）。`ring` を残すか、新しい語の既定は後で問う。
- **送り手の音型（GT）。** GT-8〜GT-11は完了した。実装は研究branch `research/gt-gesture-20261109` の `d15987b8` で、push・mainへの統合は指示を待つ。その後の未commit分は2026-11-22のS0で未検証の退避commit `17727e1d` に入れた。
  - 可聴性の門の既定化の準備（GT-7Rの指摘の修正、持続音への対応）は凍結している。自分から始めない。
  - 聴き手の側の道（NR-1の続き、PD-1の再試行）へは、自動で戻らない。
  - 変異（3b）、選択（3c）、生態が入れ替わりを自分で生むかは、作者の判断である。自分から始めない。
- **要求3のA'（RC-1〜RC-3）。** 閉じた。記憶の作り直し、身体の音高の組、身体の選び直し、聴き手の側の新しい観測は、作者が開き直すまで再開しない。
- **残した描画の記録。** `~/.cache/conchordal-test-tmp/` の `rc2/`（約3.4 GB）、`rc3/`（約3.3 GB）、GT-11の記録は、2026-11-30以降に消してよい。
- **デモ。** 11-07の基準を保って待つ。叩き台 `.orchestration/units/DEMO-B1/draft.rhai`・`DEMO-B2/draft.rhai` は素材として残るが、「動く部品でデモを組む」を進め方にしない。作者がRP-1で「外す」を仮のゴールとして採ったので、GTとFloraで一曲を作る（基準は下げず、試聴一回）。
- **browserの版（WP）。** WP-1〜WP-4は統合済み（`317e7931`）。WP-5・WP-5R（ページ）は完成し、公開は止めている。残るのはWP-6（対象機の測定）である。

外部負荷は続く前提である。計時の合否は登録済みのABBAと負荷記録を使い、静かな窓を待つ方式へ戻さない。
別の単位がtiming lockを保持している間はcargo・test・renderを起動しない。既存の取得を勝手に停止・移動・再実行しない。

## 作者へ返す判断と再開条件

**最初の作品の作曲の前の判断を作者に返す。** 作品の約束、入れ替わりを使うか、宣言する量。値や要素を試聴で選ばせない。試聴は、答えで次の作業が変わる時だけ、適用される形で一度だけ行う。

**C1の試聴は保留を維持する。** Three Windows seed 735は両版とも疎で判定しにくく、Undergrowthは拍同期Floraの身体経過が未支持で中立になるため、規則を判別できなかった。Open Time（C1-FB）では、両版とも基底のVoiceの動きがランダムに聴こえ、判別できなかった。
Voiceが厚く鳴り音楽として成り立つ場面、または拍同期Floraの機会が支持された構成で、一度だけ行う。
Voiceの動きと音程の修理が統合された後に、条件を満たすかを確かめる。実装は既定0で残すが、追加した作者規則の受入完了とは記さない。
材料と制約、宣言負荷での大きな費用増は§9.3.124にある。

**版の判断は次版について行う。** V-050の記録ではv0.5.0 Alphaは2026-10-18にtag・Pre-release公開まで完了した（`310d064e`）。
`CHANGELOG.md` のUnreleasedと現行計画の受入範囲を材料に、次の番号と公開を作者が決める。番号・Releaseの新しい承認を補わない。

**V5は別の承認を得てから行う。** 実機でPipeWireを再起動・停止し、復旧とEnd／Ctrl-Cの終了を確かめる取得は未実施。
V6/V8では代替できず、実施の判断と手順はendless設計§9・台帳§9.3.118を使う。

## 採択済みの境界と、再開しない待ち

- A1は既存の記憶を用途別に使う判断であり、文化的な選好そのものの学習や統一のなじみscoreは追加しない。
  B1は終止を観測に留める判断。B2の生成接続は別の保留であり、再開する場合は既存の評定・fit／校正資産の適用可能性を調べるところから始める。
- FEP-1・DES-1・DES-2・DES-3Aは設計と独立レビューまで。実装はGUI用のE0だけで、流れの帰属・遅いposterior・E1/Gは先取りしない。
  Listener Meter、Meter／Harmonyの二つの円、GUI時のhabitat支持、E0詳細のhoverは `41133692` まで統合済みである。
- E-11のOpen TimeとCopper and Glass旧版の採択は維持する。自己除外の技術判断は実PCM方式へ進んでおり、T-31の方式比較の試聴へ戻さない。
  samplesの手直し停止、T-28の私有の運動の型の延期、E-10の人為的な根音移動の不採用も維持する。
- 先行HVの縦・時間の評価、VR-1の作曲者の語、FA-1の振幅追従は採択済み。C1の保留で未採択へ戻さない。
  FR-1の混合RによるFlora成長則、固定調profileのTK、旧情報地形A-3をmainの採択済み機構に数えない。
  RN-1・RH-2・CAND-1などの保留は[現行計画](plan-current.md)に記す。古い単位行の「作業中」だけを根拠に再起動しない。

## 過去の再開記録

今回保存した旧本文は[2026-11-03更新前の記録](plan-history-20261018-20261103.md#saved-next-session)、それ以前は[再開記録の履歴](next-session-history.md)にある。
古い待ちや測定前の状態は当時の記録として読む。最新の作者判断と、固定したsourceに対応する取得を優先する。
