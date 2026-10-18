# 次のセッションの始め方

現在は曲の段。目標はbetaデモであり、samplesの手直しは2026-10-16に作者が停止した。
現行の範囲・単位・残る作者判断は[現行計画](plan-current.md#current-stage)を読む。
この再開点は2026-10-18のE-11・T-30〜T-34までの単位記録と作者判断を反映している。

## 直近の再開点

T-30は作者が「Bのほうがよい」と回答済み。デモの下書きでは主旋律だけ`leave_self_out(false)`とし、全声部へ広げたCは採らない。
E-11の回答は「1：open 2:いいえ」。Furnace and Braidの次の基準は、床なし・主旋律だけ自己包含のv2に80〜112秒だけ全声部flowを置いたOpen Timeである。
Copper and Glassは床なしのv2へ差し替えず、根音の床あり・動く声部が自己除外する旧版を残す。この違いを一般規則にしない。
E-9の「3分間同じ拍で退屈」への後続はこの比較であり、旧二版への回答待ちには戻さない。

作者は同日「手法の問題なら技術的に決められないの？」と指示した。自己除外の新旧実装を試聴で選ぶ問いは取り下げた。
T-32でB-1の参照を移植し、mixから自己PCMをNSGT前に引く描画由来の参照に対し、旧approx・旧exact・T-31候補を同じgateで比較した。対象は15 family・84場面である。
T-33は、参照候補が既存同点幅1e-6内なら選択先の一致を問わず推定側の場の平坦性を確かめる規則と、motionおよび静止部分音の身体投影の修正で、全15 family・84場面を通過した。
音高移動の自己除外は所有率方式一つとなり、`leave_self_out_mode`を削除した。

T-34で現在位置の点score／levelと身体score／levelも同じ同点幅で全84場面が悪化なしとなった。
`viability_scope("environment")`の継続回復得点を同じ所有率方式へ揃え、本番の旧近似と`leave_self_out_harmonics`を削除した。
適合範囲はB-1の既定Tone経路・15 family・84場面であり、詳細は[自己除外の適合検査](../design-notes/loo-conformance.md)を読む。

音が変わるsamplesは13〜16（`leave_self_out(true)`による音高移動）と08・10・12（environmentの生存選択）である。作者の試聴判断を求めず、T-31の比較試聴へ戻さない。
残る作者判断は、次の版の番号とReleaseだけである。CHANGELOGはUnreleasedのまま。ここで番号や公開の承認を補わない。

T-28の私有の運動の型は、形成・区切り・変奏を原理から導けない設計問題として返されたままである。
作者はL3の最終形を曲の段の後へ延期している。E-11の回答を実装開始の指示とは読まない。E-10の人為的な根音移動も不採用のままである。

## 先に読むもの

- [現行計画](plan-current.md)。E-5〜E-11・T-28〜T-34の要点と判断の順序。2026-10-17の全体改善は完了済み。
- 台帳[§9.3.91の終了判断](../design-notes/technote-ledger.ja.md#mechanism-completion-author-closure-20261014)。機構の完成は閉じている。既定無効経路の実機再取得を待つ状態には戻さない。
- 台帳[§9.3.92](../design-notes/technote-ledger.ja.md#sample-rounded-constants-generations-endurance-20261014)。S-2・S-3は作者採択済み、S-4のsample反映も済み。
- [文書の入口](../README.md)と[設計ノートの索引](../design-notes/README.md)。関連する設計と記録を探す場合に使う。

## 過去の再開記録

旧本文は[再開記録の履歴](next-session-history.md)へ原文のまま移した。
2026-10-07の開始手順、機構の完成を待っていた時点、曲の段のcからbへの移行、samplesの聴き直しは、
それぞれ当時の記録として読む。統括・workerの起動と作業規則は現在の`AGENTS.md`を確認する。
