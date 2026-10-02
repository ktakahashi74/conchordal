# 身体評価runtime観測: 途中出生と退役の結果

日付: 2026-09-27。隔離worktree `.worktrees/body-fitness-runtime` の通常 `conchordal-render` binaryを新規結合試験から実行した。[登録](body-fitness-runtime-lifecycle-registration-20260927.md)の初回SHA256は `a002645ef5f17e136b2f4a5f5f3b60eb8e99ae9982413b705c9fa1000453e119`、診断後の最終SHA256は `d046a58e0ea80dda929508582ede92a6739c626d41f5e920fc4ea774305c4831`。最終試験sourceのSHA256は `fbae40a21c08bc11493b37fd359fb5bef595b9fe95d6a719140ead58787d8d6f`。

frame 48はsource 1のみを受理。frame 96はsource 1と途中出生したsource 2を受理し、source 2の出生sampleは29,184（0.608秒のhop開始）だった。frame 144ではrelease後のsource 1がまだ生存し、death recordは1.696秒。frame 192ではsource 1が観測source一覧から外れ、source 2のみを受理した。source 2の出生sample 29,184とgeneration 0は維持された。三つの検査対象frameは各sourceの支持終端がbatchと一致し、support/receive/decisionの順序と最大age 4,800 sample以内を満たした。

frame 192のsource 2自己除去環境massは `1.6286016589137087e-12`。正値だが非常に小さい。通常reportだけでは退役sourceのrelease PCMがその時点で実在したことを確認できず、tail由来の質量とは解釈しない。renderer単体には退役tailを共有環境へ残す検査があるが、この通常render試験は出生・退役のsource identityと受理時刻を検証する範囲。世代変更や同じidの再利用は操作していない。

初回targetedは試験closureのRust lifetimeでコンパイル失敗。次の実取得はSeq Voiceのrelease前自然死で共存条件に失敗。endurance固定後の取得はframe 144で未退役と判明し、退役後の観測点をframe 192へ修正した。最終targetedは1成功・0失敗、`cargo fmt --all --check`と`git diff --check`も通過。最終ログは `target/body-fitness-runtime-lifecycle-20260927/targeted.log`、同一shellの終了値は同ディレクトリの `targeted-status.txt`。全suiteは統合担当の検査対象。
