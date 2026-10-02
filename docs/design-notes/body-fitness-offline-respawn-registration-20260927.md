# 通常offlineの親付きrespawn: 第十七版の事前登録

登録: 2026-09-27。第十六版dの410 source（manifest SHA-256 `01b8cc1c04aa9ffc2d79fc015f74852f7fb6d488215bff934a4394b8b4fddcaa`）を `.worktrees/body-fitness-respawn` へ内容照合して引き継いだ。以下を実装・音声取得前に固定する。背景死亡の乱数だけを既存ソースの算式から独立に先行計算した。音声、身体評価値、energyや出生結果はまだ取得していない。

## 対象と固定条件

通常の決定的 `conchordal-render` の親付きPeakBiased respawnを身体代謝と結合する。新しい `body_fitness_respawn_offline` flagは既定falseで、`body_fitness_metabolism_offline = true` との併用時だけ許可する。初回Field出生、action、observationとの併用、instrument、動的解析や制御更新は拒否する。既存単独flagの受理範囲は広げない。対象は時刻0の同一PopulationのEntrain founder 3声と最初のrespawn一機会。今回の固定sceneは次とする。

```rhai
seed(7);
place(
    harmonic().brain("entrain").sustain().anchor().brightness(0.7)
        .endurance(2.0).recovery(10.0)
        .attack_cost_fraction(0.0).attack_recharge_fraction(0.0)
        .amp(0.06).send(habitat_bus | presentation_bus)
        .respawn_consonance().respawn_capacity(3)
        .respawn_min_c_level(0.0).respawn_background_death_rate(0.5),
    line(380.0, 520.0).count(3));
flush();
wait(1.6);
```

48 kHz、512 samples/hop、既定解析空間55–8000 Hz・96 bins/oct、既定kernelとhabituation無効。Linear配置は380/450/520 Hzで、respawnの許容範囲も380–520 Hz。共通specのanchorにより親と子の出生後周波数は固定し、この一機会で移動の効果を混ぜない。身体はHarmonic、brightness 0.7、両bus amp 0.06。非ゼロ代謝係数はendurance 2秒・recovery 10秒、attack cost/rechargeは0。別の環境Voiceは置かない。

PeakBiased既定値を固定する: candidate_count 16、proposal_sigma_st 9、same_band_discount 0.08、same_band_window_cents 35、octave_discount 0.20、octave_window_cents 35、local_search_radius_st 0.50、local_search_step_st 0.05、scene_score_exponent 0.35。全binの正の身体scoreから既存peak抽出と不足時の順位補充を使う。Field density massと混同しない。最低levelは本条件0、高閾値拒否対照は1とする。

背景死亡は `DefaultHasher` へscenario seed・current_frame・substep_idxを順にHashした値と `0xBADC_0FFE_E0DD_F00D` のXORをSmallRng seedとし、64 samplesごとの8 substepで生存・未removeのVoice順に `random::<f32>() < rate * dt_step` を判定する。RNG-only計算は最初の単独死亡をframe143/substep0、Voice 3（hop開始sample73216）、次の背景死亡をframe154/substep0と予測した。frame143のcleanupで子ID4を追加した場合の計算である。Finishは1.6秒、frame150付近であり、子のframe144初回代謝と観測の後、二度目の背景死亡の前に置く。実際のFinish dispatch frameは取得で記録する。

この予測は他の自然死亡がなく、founderが最初の背景死亡まで生きる条件付きである。自然死亡、死亡時刻の不一致、同時複数死亡、二回目respawn、親不在は取得の失敗として保存し、結果を見てseed・率・Finish・係数を動かさない。RNG helperとrawは `target/respawn-validation/turnover_rng_only.rs`、`.raw` に保存する。

## 実装する因果順序と所有権

各hopでは主解析受理とhabituation適用後、完全な自己除去batchを各既存Voiceの代謝へinstallする。phonation収集時のonset代謝の後、各substepの背景死亡判定とlifecycle更新があり、その後のcleanupが生存親poolを作る。この実post-update energyだけを親重みにする。hop開始時energyも記録し、更新前energyの流用と区別する。

cleanupの一機会で既存の親重み付き抽選を一回行い、その親と実templateに対応する予定子ID、population、member_idx、系譜generation、parent_idを確定する。予定IDはallocatorと同じ未使用ID探索を非破壊で行う。`spawn_counter` は拒否でも進め、ID・memberは成功時だけ消費する。系譜generation（親+1）と身体Recipeのbody_generationは別の軸であり、新生児の候補身体generationは0とする。

この確定metadataで全許容binを72hopの代表身体から採点し、既存のpeak抽出、親距離重み、RNG消費順を保つ。局所探索で訪れる周波数と最終Hzも同じ身体で評価する。最低level拒否時は子を作らない。欠測・不一致を旧点評価や古い表で補わない。実spawn成功時は予定ID、member、parent、系譜generation、Hz、身体、Recipe identityを照合する。初回出生と共通の代表身体計算は再利用するが、親なし・世代0固定のmetadataは転用しない。既存の試験専用F4c/e hookは通常経路と分離する。

cleanupはphonation収集後なので、子の出生hop代謝receiptを作らない。cleanup後、描画前に現在の全Voice IDでself sound slotを準備し、新子に長さ512のゼロ自己PCMを確保する。既存の履歴はこの準備だけでは進めない。描画後observerは新しい全source集合を完全一致で処理し、子の履歴を当該hop以前の共有解析からcloneして `mixed - own` の当該hopを進める。翌hopに子は `birth_sample = 73216` を保持した初回SourceRemoved installを受け、実lifecycleに参加する。

退役Voiceが `should_retain()` に該当すればobserverに残り、最大4 sourceとなる。除去後もToneのtailは混合環境に残り得るが、存在しないVoiceの自己PCMとして差し引かない。source集合の変化は当該死亡Voiceの正当な退役と一子追加だけを許す。現在のallocatorは新IDを使うので、通常respawnで同ID再利用を実証したとは扱わない。

## 取得と合否

固定sceneのOFF（両flag false: 既存点代謝・点出生）とON（両flag true: 身体代謝・身体出生）を各2回取得する。これは結合全体の比較であり、代謝単独と出生単独の効果を分離する比較ではない。同mode内でWAV、respawn、親pool・抽選、全bin・局所候補、最終Hz、子の出生/翌hop遷移の対象JSONLの一致を要求する。mode間の選択親、最終Hz、WAVが異なることは要求しない。

ONでは最初の単独死亡が登録frame143/Voice3、親poolが残る二声の実post-update energy、成功子がID4・member3・親+1世代であることを要求する。少なくとも一親の実energyがOFFとの差 `1e-6` を超え、少なくとも一候補の身体scoreと共有点scoreの差が `1e-6` を超えることを要求する。差が出なければ固定条件不達と記録する。出生hopの子の代謝receiptはnull、次hopはSourceRemovedで正しいsource identity・支持・epoch・空間を持ち、実energy更新に使われることを確認する。

独立対照では、記録した親pool全件とspawn seedから別RNGで親を再抽選する。親重み、peak候補抽出、局所候補と実選択、最低level、選択前後RNGを照合する。少なくとも一binと最終Hzは別Tone・別Analysisの72hop密度と直接積分から身体score/levelを照合する。この独立数値検査と通常rendererでの実消費を結果文書で分ける。ONの高閾値1の対照は拒否し、spawn_counterだけが一回進み、ID/member/Voiceは増えないことを確認する。

必須負例はシナリオの誤構成とinstrument/他flag併用、親不在・同時複数死亡・二回目機会・5 source目、想定外追加/削除/世代・活動中ID衝突、更新前energy流用・親poolや選択親の差替え、template/予定ID/memberの変更、候補表と実子の不一致、worker欠測、支持の未来/期限切れ、epoch/space/habituation不一致、自己PCM欠落・511 samples、翌hopの子source欠落。既存試験が証明する範囲、新たな局所fault injection、通常rendererでの拒否、runtime assertionだけの境界を別々に記録し、assertionの存在だけを注入済みとは呼ばない。

source固定後にfmt、標準Clippy、全target check、`RUST_BACKTRACE=1 cargo test -- --nocapture` を行い、stdout/stderrと同shellの実exitを保存する。source manifest・binary hash・scene/config hash・raw取得を対応付け、失敗履歴を残す。性能の専有取得は今回に含めず、第十五版の計時を転用しない。長期生態、非同期代謝の公平性、音色genotypeの継承・変異、同ID再利用、実device、作者採用、F4/F5/F6全体は未完了のまま。main採用、既定変更、commit、pushは含めない。
