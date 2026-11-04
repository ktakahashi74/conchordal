# 次のセッションの始め方

2026-11-05更新。実装は `origin/main` の `3c3f15f4` まで、経過は同日のDEMO-B1までの統括記録を反映した再開点である。
曲の段の目標はbetaデモ。新しいセッションで、古い「自己除外の比較試聴待ち」「残るのはv0.5.0公開だけ」へ戻さない。

## 最初に読むもの

1. `AGENTS.md` と[現行計画](plan-current.md#current-stage)。承認された範囲、既定値、作者判断と科学的受入の区別を確認する。
2. 台帳[§9.3.123](../design-notes/technote-ledger.ja.md#harmony-decisions-20261103)・[§9.3.124](../design-notes/technote-ledger.ja.md#harmony-action-consequences-20261103)。A1/B1/C1は採択済み、C1は実装済み・作者試聴保留という順序を読む。
3. 台帳[§9.3.117〜118](../design-notes/technote-ledger.ja.md#incremental-endless-scores-20261031)。endlessの実装と、IP-ACQの受入を読む。V5未取得、V6合格、V7取り直し中、V8の差分推定と実機再取得を混ぜない。
4. 台帳[§9.3.126](../design-notes/technote-ledger.ja.md#voice-self-exposure-one-listener-20261104)・[§9.3.127](../design-notes/technote-ledger.ja.md#voice-melody-stop-20261105)と[現行計画のVoiceの節](plan-current.md)。VB-3は統合済み、TU-2／TU-3は統合しない。Voiceの旋律の生成は停止し、デモは方向Bで進める。
5. 台帳[§9.3.125](../design-notes/technote-ledger.ja.md#listener-prediction-e0)と[Listener Twin Design](../design-notes/listener-twin.md#gui-1-presentation-prediction-and-current-display-contract)。E0は未較正の混合音響baselineであり、DES-1／DES-2全体の実装ではない。後続のGUIの到達点は `41133692`。
6. [統括とhostの運用](../operations/orchestration.md)。単位記録の所在、既存job、timing lock、再取得の制約を確認する。必要な設計は[文書の入口](../README.md)・[設計ノート索引](../design-notes/README.md)から探す。

## 進行中の長い取得を先に確認する

**ここに記す稼働状態は統括記録に基づく。新セッションでは実際のserviceと最新出力を確認する。**
DOC-3からはuser busへ接続できず、実際の稼働・終了・freeze状態は確認できていない。
既存取得の再起動や、別のsourceでの取り直しを始める前に、固定sourceとbinary、既存出力、真の終了状態を確かめる。

- **IP-ACQ V7：推定器の精度（取り直し中）。** 最初の取得 `conchordal-ipacq-v7-20261102` は2×10⁷ checkpointまで全件合格したが、
  2026-11-04 03:32のuser systemdの終了で約2.64×10⁷発行のときに止まった。同じ固定binary（`IP-ACQ/pins/v7`）で0から取り直している
  （`conchordal-ipacq-v7-r2-20261104`、label `full-r2`、出力 `IP-ACQ/v7/full-r2-*`、凍結のguardも再起動済み）。
  次は3×10⁷ checkpointと全出力のfailed／inconclusive／uncertified、凍結、真のexit、source対応を回収し、IP-ACQの報告へまとめる。
  「途中で失敗0」を最終精度合格とはしない。旧IP-V7／V7bの旧土台や中断分を統合版に数えない。
- **IP-ACQのV6・V8は取得済み。** V6は譜面132,000秒を合格、V8は計測交絡の限定切り分けと疑い2条件の実機再取得まで記録済み。
  V7が動いていることを理由に、V6/V8も未取得へ戻さない。V7の修理が必要になった場合は、登録条件に従って影響する取得を決める。
- **方向Bのデモ。** 統括の叩き台は `.orchestration/units/DEMO-B1/draft.rhai`（記録は同じ単位のreport）。部品と形は作曲者の量であり、試聴で選ばせない。
  Voiceの旋律の生成は作者の規則で停止した（PD-1／PD-2、§9.3.127）。同じ問題へ別の入口から自動で戻らない。

外部負荷は続く前提である。計時の合否は登録済みのABBAと負荷記録を使い、静かな窓を待つ方式へ戻さない。
別の単位がtiming lockを保持している間はcargo・test・renderを起動しない。既存の取得を勝手に停止・移動・再実行しない。

## 作者へ返す判断と再開条件

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
