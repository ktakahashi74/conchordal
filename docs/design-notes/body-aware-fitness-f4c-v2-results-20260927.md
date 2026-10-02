# F4c v2: 身体評価から親energyとHereditary出生への一機会接続

日付: 2026-09-27。[取得前登録](body-aware-fitness-f4c-v2-registration-20260927.md)に基づく隔離検査。対象は `.worktrees/body-fitness-lifecycle` の `cfg(test)` 経路であり、検査後に隔離した[統合第八版](body-fitness-integration-validation-20260926.md)へ取り込んだ。mainへの採用は行っていない。旧[F4cの不合格](body-aware-fitness-f4c-results-20260927.md)とそのcapsuleは変更していない。

## 親の代謝と出生

Sustain templateをendurance 10秒、recovery 1秒、dissonance penalty 1に固定し、実Voiceのcommitを0.01秒で一度実行した。親1のenergyは0.350000から0.355359、親3は0.650000から0.655280へ変化した。身体levelはそれぞれ0.605577、0.598094であり、登録した代謝式の期待energyと今回のf32値は一致した。内部attackは0件だった。

通常の親poolにこの更新後energyが入り、既存のenergy比例抽選が親1を選んだ。親1の重みを正規化した確率は0.350000から0.351618へ変化した。固定seed 7の一回の抽選結果は旧試験と同じであり、選択親の交代を示したとはしない。独立再演の親抽選、候補16 slot、終了RNGと実経路を照合した。親抽選前後のprobeは独立再演の記録、候補選択後のprobeは実経路と参照の双方を保存した。

身体による候補採点から子周波数bit `1134839946` を選び、閾値なしと登録式の低閾値0.247330では出生、高閾値0.879641では拒否した。出生後の親id、子generation、body/freq/modulator、宣言代表recipeの同一性、id/member index/eventと死亡再処理の境界を確認した。通常発声のToneSpec全欄との一致や音色遺伝の証明ではない。

## 対照と失効

身体評価を使う二試行は、旧点Cを低値と高値へ変更しても、保存した親energy・親抽選・候補・子・終了RNG・イベントの全recordが一致した。一方、身体contextを外した旧点C経路では、低値の親energyが0.349525/0.649525、高値が0.358975/0.658975となり、それぞれ登録した式に一致した。出生閾値0.5で低値は拒否、高値は出生した。

親generation、live template brightness、予約child id、frame、epoch、候補slot順序を一つずつ壊した六条件は、通常cleanup_deadから試験用の照合へ到達して拒否された。子を追加せず、id/member indexを消費せず、身体levelの読出しにも進まなかった。これは試験専用assertionの検証であり、通常runtimeの失効処理・回復保証ではない。

## 検証と保存

取得前登録のSHA256は `72093dbc815e25b47c9f5991c652920e840611ae0f12bb0cf56859d2d650667f`。`target/body-fitness-f4c-v2-capsule-20260927/` に変更前source、登録、差分、各実行ログを保存する。最初の二回はJSON macro記法とimport可視性のコンパイルエラーで、数値試験へ到達しなかった。修正は記法とimportに限定し、登録した係数・seed・音源・閾値は変更しなかった。三回目のtargeted検査は通過した。その後、点C毒入れの比較を全保存record一致へ強め、RNG欄を参照probeと明記して全suiteを実行した。

隔離worktreeの全cargo testは2026-09-27 11:11:35 JSTにexit 0、1205成功・0失敗・40 ignoreだった。fmtと標準Clippy、release buildも通過した。通常release renderは旧F4c capsuleのbinaryとbyte一致し、SHA256は `396ec45c4c180b35f681f20df6d2f74a08a1c1164d26f0c706434d3e8f9b8475`。同一binaryなので旧登録の固定None回帰（WAVと非timing 2319記録一致）を対応づけ、新しい回帰取得は行わなかった。全suite内のrender結合テストは今回も実行した。

この結果が示す範囲は、固定した一機会における身体評価→実energy変化→親重み→Hereditary候補採点→子の生成である。通常runtimeへの常時接続、出生の残る入口、非同期配送、音色遺伝、実時間の有効評価率は未完了のままである。

統合第八版では1243成功・0失敗・43 ignore、fmt・標準Clippy・全target checkを通過し、F4c結果JSONは隔離版とbyte一致した。通常release binaryは統合前と異なったため固定4条件を再取得し、全条件のWAV・登録対象記録の一致を確認した。統合初回の古いtest binary再利用と、再ビルドで検出した試験ヘルパーの可視性不足は記録を保存し、最終結果から除外した。修正後の全suiteにF4cの成功行が存在することも照合した。
