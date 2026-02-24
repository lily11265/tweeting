# BirdNET bridge 통합 테스트
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

def main():
    from birdnet_bridge import run_birdnet_prediction, _parse_species_name, _parse_time

    # 종명 파싱 테스트
    assert _parse_species_name("Otus scops_소쩍새") == "소쩍새"
    assert _parse_species_name("Poecile atricapillus_Black-capped Chickadee") == "Black-capped Chickadee"
    assert _parse_species_name("UnknownSpecies") == "UnknownSpecies"
    print("✅ 종명 파싱 테스트 통과")

    # 시간 파싱 테스트
    assert _parse_time("00:01:30.50") == 90.5
    assert _parse_time("00:00:03.00") == 3.0
    assert _parse_time("42.5") == 42.5
    print("✅ 시간 파싱 테스트 통과")

    # 예제 WAV 파일로 실제 예측 테스트
    wav_path = "birdnet-main/example/soundscape.wav"

    if not os.path.exists(wav_path):
        print("⚠️ 예제 WAV 파일이 없습니다, 예측 테스트 건너뜀:", wav_path)
        return

    print("\nBirdNET 예측 테스트 시작...")
    results = run_birdnet_prediction(
        wav_path,
        confidence_threshold=0.3,
        lang="ko",
        progress_callback=lambda msg: print(f"  >> {msg}"),
    )

    print(f"\n총 검출: {len(results)}건")
    for r in results[:10]:
        sp = r["species"]
        t0, t1 = r["t_start"], r["t_end"]
        print(f"  {sp:20s}  {t0:7.1f} - {t1:7.1f}s  conf={r['confidence']}")

    assert len(results) > 0, "검출 결과가 0건"
    assert all("species" in r and "t_start" in r and "t_end" in r for r in results)
    print("\n✅ 모든 테스트 통과!")


if __name__ == "__main__":
    main()
