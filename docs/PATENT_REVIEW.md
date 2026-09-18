# Goodsize Inc. 특허 현황 조사

- 조사일: 2026-09-14
- 출처: Google Patents 검색(assignee "Goodsize"), freepatentsonline, 저장소 `PATENTS.md` / `LICENSE-STREAMER.md`
- 이 문서는 공개 자료 정리이며 법률 의견이 아닙니다. 자체 Streamer 구현 착수 전 변리사의 FTO(freedom-to-operate) 검토가 필요합니다

## 1. 저장소가 밝힌 특허

`PATENTS.md` 와 `LICENSE-STREAMER.md` 는 다음을 명시합니다.

- Streamer(`src/avaturn_live_streamer/`)는 Goodsize 소유의 "pending and/or issued" 특허를 실시함
- 예시로 든 출원: "U.S. Patent Application No. 08915-P0003A", 제목 *System, Apparatus, and Method for Generating Avatar*. 이 번호 형식은 USPTO 출원번호가 아니라 **대리인 관리번호(docket)** 형식임
- 특허 대상으로 열거된 기능: 오디오 스케줄러, 드리프트 보정 스트림 클록, 프레임 정밀 이벤트 발행 파이프라인, 대칭 이중 스트림 렌더 조정
- 특허 라이선스는 "배포된 코드 형태의 비상업 사용"에 한정. 독자 구현·재작성·포팅·클린룸 구현에는 실시권 없음 (3-5항)
- Renderer 라이선스(`LICENSE-RENDERER.md` Required Notice)에도 동일 문구가 있으나, Renderer 관련 특허는 어디에도 특정되어 있지 않음

## 2. 공개 검색 결과 (Goodsize Inc. 명의)

| 번호 | 제목 | 출원일 | 공개/등록일 | AVTR-1 관련성 |
| --- | --- | --- | --- | --- |
| US 11,494,963 B2 | Methods and systems for generating a resolved 3D (R3D) avatar | 2020-11-09 | 2022-11-08 등록 | 없음. 사진·영상 기반 3D 신체·머리 메시 정합(In3D). 음성 구동 영상 아바타와 무관 |
| US 11,386,580 B1 | Guiding user to comply with application-specific requirements | 2021-08-13 | 2022-07-12 등록 | 없음. 촬영 가이드 UI |
| US 11,562,504 B1 | Predicting lens attribute | 2022-01-26 | 2023-01-24 등록 | 없음. 카메라 렌즈 추정 |
| **US 12,322,020 B1** (공개 US 2025/0173939 A1, 출원 18/796,304) | System apparatus and method for providing facial expression to avatars | 2024-08-07 (가출원 63/603,700, 2023-11-29) | 2025-05-29 공개, 2025-06-03 등록 | **간접 관련**. 아래 3장 |

발명자: Sergei Sherman, Dmitrii Ulianov (US 12,322,020).

## 3. US 12,322,020 B1 요지

- **청구항 1 (시스템)**: (a) 아바타 얼굴을 소스 얼굴로 교체하고 소스의 표정을 실시간으로 아바타에 전이하도록 신경망을 학습하는 학습 시스템, (b) 학습된 신경망으로 아바타가 소스 표정을 따라가는 영상 프레임을 만드는 추론 시스템. 추론은 3D 얼굴 모델 재구성 → 기하 변경으로 애니메이션 → 렌더 → 신경망 후처리 순서
- **청구항 11**: 청구항 1 의 방법 버전
- 종속항: 타깃 얼굴 DB·소스 얼굴 DB(2·12항), 표정 이미지 DB 로 표정 구동(8–10·18–20항)
- **없는 요소**: 오디오 구동 애니메이션, 오디오 버퍼·스케줄링, 클록 드리프트 보정, 프레임 타임스탬프·이벤트 발행, 화자·청자 이중 스트림, 끼어들기 처리, WebRTC

**AVTR-1 파이프라인과의 대응**: AVTR-1 은 음성 → 회전·표정 계수 → LivePortrait 워프/디코더 순서이며, "얼굴 교체(face swap)" 나 "3D 얼굴 모델 재구성" 단계가 없습니다. 청구항 1 의 필수 요소(학습 시스템 + 3D 재구성 + 얼굴 교체)를 모두 갖추지 않으므로 문언 침해 가능성은 낮아 보입니다. 다만 최종 판단은 변리사 검토 사항입니다.

## 4. Streamer 출원("Generating Avatar")의 공개 상태

- 가출원 63/603,700 의 제목이 *System Apparatus and Method for Generating Avatar* 로 `PATENTS.md` 의 출원 제목과 일치합니다. Streamer 출원은 이 가출원을 우선권으로 하는 **별도 정규출원**일 가능성이 높습니다
- 2026-09-14 기준 Google Patents 에서 해당 제목의 Goodsize 명의 공개 문서를 찾지 못했습니다. 가출원 기준 18개월(2025-05-29)이 지났으므로, (a) 비공개 신청(non-publication request)을 했거나, (b) 더 늦은 우선권의 신규 출원이거나, (c) 미국 외 출원일 수 있습니다
- 따라서 **청구 범위를 현재 확인할 수 없습니다.** 문의 메일(질의 2.3)에서 출원번호 또는 청구항 요지 제공을 요청했습니다

## 5. 자체 구현에 대한 위험 평가

| 구성 | 위험 | 근거 |
| --- | --- | --- |
| AVTR-1 모델 + HuBERT 그대로 사용 | 낮음 | Community License 로 허용. 특허 고지는 Streamer 한정 |
| 자체 렌더러 (LivePortrait 기반) | 낮음~중간 | 특정된 Renderer 특허 없음. US 12,322,020 은 얼굴 교체·3D 재구성 요소가 필수여서 구조가 다름. Required Notice 문구가 "독자 구현 불허" 를 말하지만 근거 특허가 없으면 계약상 문구에 그침 |
| 자체 로컬 재생기 (오디오 큐, A/V 동기, barge-in) | **중간~높음, 판단 불가** | 미공개 출원의 청구 범위를 모름. `PATENTS.md` 가 열거한 4가지 기능 중 "오디오 스케줄러"·"이중 스트림 조정" 은 로컬 재생기에도 유사 개념이 필요 |
| Streamer 코드 참조 후 재작성 | 높음 | 저작권(PolyForm NC)과 특허 모두 문제. Streamer 소스는 열지 않고 요구사항만으로 설계 권장 |

## 6. 권고

1. 문의 회신에서 출원번호를 받으면 USPTO Patent Center 에서 청구항을 확보하고 변리사 FTO 검토
2. 회신이 없거나 청구항이 광범위하면, 로컬 재생기는 **표준 미디어 플레이어 방식**(오디오 장치 클록 마스터, 프레임 PTS 정렬)으로 구현하고 설계 문서에 선행기술(기존 A/V 동기 특허 US 7,620,137 등)을 기록해 두는 것이 방어에 유리
3. Renderer 계약을 하더라도 Streamer 특허는 별개이므로 계약서에 "로컬 재생 구현에 대한 특허 비주장(non-assert)" 조항을 요구
4. 이 문서는 Avaturn 회신과 FTO 결과가 나오면 갱신
