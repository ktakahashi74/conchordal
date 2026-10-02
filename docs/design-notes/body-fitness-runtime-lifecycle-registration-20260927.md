# 身体評価runtime観測: 途中出生と退役の取得前登録

日付: 2026-09-27。状態: 初回targeted後の修正登録。初回版SHA256は `a002645ef5f17e136b2f4a5f5f3b60eb8e99ae9982413b705c9fa1000453e119`。対象は隔離worktree `.worktrees/body-fitness-runtime` の通常 `conchordal-render` binary。試験は新規 `tests/body_fitness_observation_lifecycle.rs` に置く。source coreと既存 `tests/render_binary.rs` は変更しない。

seed 7、48 kHz、512 sample/hop、`body_fitness_observation = true`。Sine 440 Hzの持続Voiceを開始時に配置し、0.6秒後にSine 660 Hzの持続Voiceを追加する。さらに0.6秒後に最初のPopulationをreleaseし、1.1秒後に二番目もreleaseする。両Voiceはhabitatへ送る。reportの `body_fitness_observation` を通常render経路から読む。reportの間引きは48 frameごとなので、frame 48を出生前、frame 96を共存中、frame 192を最初のVoice退役後の観測点とする。イベントはhop境界へ切り上がるため、二番目の出生sampleは0.6秒の指定tick以上で最初の該当hop開始sampleと一致することを要求し、固定値への丸めは試験結果で確認する。

成功条件: 出生前の受理recordはsource 1のみを持ち、source 2を先取りしない。共存recordは異なる二つのsource idを持ち、source 2の `birth_sample` が出生hopの開始sample、source 1の値が0である。受理recordでは各sourceの支持終端が共通batchの支持終端と一致し、`birth_sample <= support_end_sample`、`support_end_sample <= received_at_sample <= decision_at_sample`、ageは4800 sample以内。退役後recordはsource 1を生存sourceとして列挙せず、source 2の出生sampleを維持する。source 2の自己除去環境massが正であることも試験で確認する。ただし通常reportは退役sourceのPCMを保存しないため、微小な正massをrelease tail由来の証拠とは扱わない。generationは通常spawnの0を確認するだけで、世代変更は操作しない。

拒否recordと一時的な未到着recordはその理由を独立に扱い、受理recordだけを数値条件へ使う。中間の出生・退役で一時的なIdentityMismatchが出ても、後続の受理回復を要求する。WAVの比較や壁時計費用測定はこの一試験に含めない。targeted cargo testの初回結果を保持し、後続の全suiteは統合担当が実施する。

修正理由: 初回は試験closureのRust lifetimeコンパイル失敗。次の実取得では `brain("seq")` の最初のVoiceがrelease予定前の1.0027秒に自然死したため、固定 `brain("drone").endurance(10)` へ変更した。その次の取得では最初のVoiceのrelease後の死亡は1.696秒で、frame 144の時点ではまだ生存。したがって退役後の観測点をframe 192に変更し、二番目のVoiceをそこまで保持する。frame 192のmassは約1.6e-12と小さく、物理的なtail存続の証拠に使わない。これは初回版の前提修正後の取得であり、完全な初回事前登録として扱わない。
