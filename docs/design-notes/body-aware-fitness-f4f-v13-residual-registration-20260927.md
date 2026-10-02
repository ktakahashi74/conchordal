# F4f 第十三版: 初回 Field spawn 残余試験の固定登録（2026-09-27）

対象は `.worktrees/body-fitness-metabolism` の第十三版隔離sourceにある `cfg(test)` の F4f オフライン注入である。通常sourceの出生配線は追加しない。試験本体は `src/life/community/actions/f4f_offline.rs` だけを変更する。既存の[第十二版結果](body-aware-fitness-f4f-initial-results-20260927.md)に列挙した未取得条件を検査する。取得後に失敗したfixtureを成功条件へ調整せず、失敗理由と改訂が必要な箇所を結果へ残す。

基準fixtureはCommunity seed 7、48 kHz、512 samples/hop、frame 64、population ID 7、子ID 2・3、初期range 380–520 Hz、Log2Space 80–4000 Hz・48 bins/octave、代表72 hop、解析epoch 37、Harmonic bodyのbrightness 0.7、inharmonic 0、unison 1。最初の環境は既登録の440 Hz・振幅0.06のSineを72 hopとする。比較用の環境だけ、440 Hzと466.16 HzのSineを各振幅0.03で同じ72 hop合成し、`consonance_density_roughness_gain` の `rho` を0または3.0へ固定する。身体比較は同じ子specの `BodyMethod::Sine` と `Harmonic`（後者のbrightness 0.7）を使う。解析パラメータを通してLandscapeの有効massを再計算し、注入tableを各条件で準備する。

判定条件は以下のとおり。

1. Dissonance/Edge、それぞれPeakとDensity、spacing 0、tension 0を通して2子ずつ出生させる。Peakは候補 `fitness.level` の最小値または0.5への最短距離から、同値時は低いbinを独立期待binとする。Densityは候補ごとの非負massを独立 `WeightedIndex` に与え、seed 7の最初の子の選択binと照合する。2子の最終周波数・identity・runtime event属性も確認する。
2. 実身体・rho比較は、同じ候補binごとの `consonance_mass` の差と、全binで正規化した抽選確率ベクトルの全変動距離を判定する。Sine対Harmonicはrho 0で比較し、rho 0対3はHarmonic固定で比較する。各対で少なくとも1binのmass bitが異なり、全変動距離が `1e-5` を超える条件を合格とする。rho対照では同じ候補のfitness score bitが一致することも要求する。身体のfitness scoreだけをmassの代理にしない。bin選択は各mass列を独立 `WeightedIndex` に与えて固定seedと照合する。数値が変わらなければそのまま失敗として記録する。
3. 注入経由の狭いrangeは220.0–220.1 Hzと408.0–420.0 Hzを固定する。PeakとDensityの両方で全候補slotのbin番号・clamp周波数を照合し、2子の最終周波数が要求range内にあることを確認する。
4. 380–520 Hz、Density、spacing 0.25 ERB、tension 0.4の2子で、連続jitter後の最終周波数間距離が0.25 ERB以上であることを確認する。runtime eventはID、population、member、周波数bit、時刻、世代0、parentなし、Initial理由をVoiceと照合する。
5. 同じseed・spec・strategy・環境で、注入contextを置かない初回Spawnを2回独立に実行し、周波数bitとruntime event属性の一致を確かめる。これは試験buildの既定OFF経路の再現性検査であり、製品runtimeへの注入ON/OFF比較ではない。
6. Gap/Uniformの `Context::prepare` は明示拒否する。注入contextなしの既存経路では同じseedの結果が再現し、指定range内へ出生することを確認する。Gap/Uniformを身体評価済みとして数えない。
7. 全bin占有時の一様fallbackはspacing 100 ERB、同じ子ID 2・3、Community seed 7–14の8固定例で確認する。各例で後続子の選択binが準備済みrangeに入り、8例全体で2種類以上のbinが選ばれることを要求する。8例は分布推定ではなく、単一点への退化が起きない境界検査として扱う。

合格範囲は個々の固定fixtureとassertionに限定する。狭域外の一般保証、分布の統計的収束、非同期通常runtime、実時間費用、長期選択効果は含めない。focused試験だけを実行し、full suiteとtimingは統合担当へ渡す。
