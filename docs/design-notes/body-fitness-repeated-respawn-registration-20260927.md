# 通常offlineの反復親付きrespawn：第十八版の事前登録

2026-09-27。第十七版fの固定414 source（manifest SHA-256 `22791e93aac37af82c606877d4fb5f17f68079e8fcb223aa46192cc128a58ae9`）から、通常offlineの二度目の親付きrespawnとsource交代を検証する。作業先は `.worktrees/body-fitness-repeated-respawn`。音声、身体score、energy取得前に本条件を固定する。第十七版の一機会sceneと拒否対照は回帰試験として維持する。

## 固定sceneと事前予測

第十七版のsceneから終了時刻だけを1.6秒から1.72秒へ延ばす。

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
wait(1.72);
```

48 kHz、512 samples/hop、8 control substeps、解析空間55–8000 Hz・96 bins/oct、habituation無効、既存PeakBiased係数は第十七版と同じ。新flagは追加せず、`body_fitness_respawn_offline` と `body_fitness_metabolism_offline` を両方falseにするOFF、両方trueにするONを各2回取得する。移動や観測flagとの同時ONは扱わない。

`target/repeated-respawn-validation/turnover_rng_only.rs` は音声を生成せず、既存SmallRngと背景死亡率0.5/秒で予測する。初期ID1/2/3が生存し、死亡後に子を一声追加し、他のlifecycle死がない条件で、死亡はframe143/substep0/ID3、frame154/substep0/ID2、frame225/substep6/ID5。最初の二機会の間に第一子ID4の初回receiptが入る。Finishはframe162付近（登録判定範囲156以上225未満）で実dispatchを記録する。実lifecycleでこの生存条件を満たさなければ固定scene不達として残す。seed、死亡率、energy係数を取得後に調整しない。

## 実装契約

機会を連番、決定sample、死亡sourceの完全identity（ID・系譜generation・birth_sample）、spawn sequenceで識別する。一回だけの恒久的なattempted状態を機会状態に変更し、前機会のcandidate、予定子、親pool、pending reportを次へ持ち越さない。同一hop内の二重機会、未排出record、前児の翌hop検証が終わる前の重なりは拒否する。productionに「ちょうど二回」やframe143/154の特例は加えず、登録sceneで二回を検査する。

各機会で更新後の実親energyから親を抽選する。第一poolはID1/2、第二poolはID1/4。第二poolのID4は第一子の実出生sample73216・系譜generation1を保持する。第二選択親が誰になるか、第二子のHzやenergyは合否条件として固定しない。第二子はID5、member4、選択親の系譜generation+1、出生時body generation0。親の身体を遺伝する実装ではなく、固定Population templateから子を構築する。

出生前の全source集合とbatchを完全照合する。保持中の死者も4source上限に含める。前後集合の差は実退役と今回の新子だけに限り、birth_sampleを初期founderの0へ戻さない。第一機会はsource1/2/3→1/2/3/4、第二機会は1/2/4→1/2/4/5を期待する。退役後の集合は順に1/2/4、1/4/5。実際の保持期間が異なれば相違を記録し、根拠なく集合を間引かない。

各出生hopの子は代謝receipt不在、自己PCM512 samples全ゼロ。翌hopで正しいsource identityとepoch0、判断sampleに対応したSourceRemoved receiptを取得し、実energy更新に使う。二度目の遷移報告で第一子をbirth_sample0へ戻したり、第一子のreceiptを第二子へ割り当てたりしない。各機会の出生・翌hop・自己PCM・拒否reportを同じ機会キーで結ぶ。記録は機会ごとに排出し、候補列を無制限に保持しない。

## 取得と判定

ON/OFF各二回でWAVと対象JSON recordの同mode内一致を要求する。両modeで二死亡・二実出生を要求し、ONの機会1/2を独立に再計算する。親poolからのRNG抽選、記録済全binからのpeak抽出、候補重み付き選択、局所選択、最終Hzと実子の一致を照合する。身体score自体の独立Tone積分は先行の別環境fixtureと区別し、今回の通常sceneの全scoreを独立再現したとは扱わない。

各機会で少なくとも一候補の身体scoreと点scoreの差が1e-6を超えること、出生hopから翌hopへ子energyが更新されることを要求する。第二poolは第一子の現在energyを使用し、初回energyや前機会のcacheを使わない。ON/OFFで親、Hz、WAVが異なることや系譜generation2の子が選ばれることは要求しない。代謝と出生の結合処置であり、効果の単独分離とは呼ばない。

第十七版の未注入境界を今回のruntime検査へ追加する。正当な退役以外の削除、予期しない追加、活動中ID衝突、birth_sample/系譜改竄、自己PCM欠落・511 samples・非ゼロ、翌hop子source欠落を対象とする。評価器には同じ支持時刻での機会再利用、前機会候補の持越し、未排出record、二機会の記録混線、支持・epoch・space・habituation差を注入する。通常render、production関数への局所注入、旧試験専用hook、assertだけの範囲を分けて記録する。登録した項目が未検査なら未完と明記する。

source固定後にfmt、標準Clippy、全target checkと `RUST_BACKTRACE=1 cargo test -- --nocapture` を行い、stdout/stderrと同shellの実exitを保存する。source capsule、binary hash、scene/config、raw、失敗履歴を対応づける。実時間の専有取得、異種身体の比較、actionとの同時接続、音色遺伝、同ID再利用、main採用、既定変更、commit、pushは本単位に含めない。F4/F5/F6全体は引き続き未完了。
