# Avaturn(Goodsize Inc.) 상업 라이선스 문의 초안

- 작성일: 2026-09-14
- 수신: hello@avaturn.me (LICENSE.md 에 명시된 단일 문의 창구). 병행으로 https://avaturn.live/licensing 의 "Talk to Sales" Typeform 제출 권장
- 목적: **모델 단독**과 **모델 + Renderer** 두 가지 견적을 동시에 받고, 자체 구현 시의 권리 범위를 서면으로 확인
- 발신 전 확인: 아래 `[ ]` 표시 항목(회사명, 매출 구간, 고객사 수, 설치 대수)을 채우세요. 매출은 계열사·자회사 합산 기준입니다

---

## 영문 본문 (발송용)

Subject: Commercial licensing inquiry — AVTR-1 on-premise deployment (model-only and model + Renderer quotes)

Hello Avaturn team,

We are [ ] (company), based in [ ] (country). We are evaluating AVTR-1 for an **on-premise, single-PC deployment at our customers' sites** (kiosk / reception style: one avatar instance per PC, no public web streaming). Our consolidated annual revenue, including affiliates, is [below / at or above] USD 10M. We are not building an avatar platform or API that competes with Avaturn.Live; the avatar is a component inside our own [ ] product.

We have read LICENSE.md, LICENSE-MODEL.md, LICENSE-RENDERER.md, LICENSE-STREAMER.md and PATENTS.md in the `avaturn-live/avtr-1` repository. To decide between licensing your components and building our own, we would like to request the following.

**1. Two quotes**

- (A) **AVTR-1 model only** — if our revenue is below USD 10M this should be covered by the Community License; please confirm, or quote if a Commercial Use Agreement is still required for our case.
- (B) **AVTR-1 model + Avaturn Renderer Commercial License** for on-premise use. Please indicate the pricing structure (per deployment / per PC / per year / revenue share) and what the fee covers (updates, support, TensorRT engine builds).

We do **not** need the Avaturn Streamer as distributed: our output is a local display on the same PC. Please also quote (C) Streamer Commercial License separately in case the terms make it attractive.

**2. Clarifications on scope (written confirmation requested)**

- 2.1 Every `.py` file under `src/avtr1_renderer/` carries the header `SPDX-License-Identifier: LicenseRef-AVTR-1-Community`, while LICENSE-RENDERER.md states the whole directory is PolyForm Noncommercial. Which governs? In particular, may the **model-calling code** (`avtr1_motion_generator.py`, `models/avtr1.py`, `components/hubert.py`) and the Community-licensed `scripts/build_avtr1_engines.py` be used commercially under the Community License, independently of the Renderer license?
- 2.2 If we keep the AVTR-1 weights and HuBERT encoder but build **our own rendering stage** (LivePortrait-based, MIT) and our own local playback/scheduling code, does Goodsize assert any patent or license claim over such an independent implementation? PATENTS.md limits its scope to the Streamer; please confirm that no Goodsize patent rights are asserted against an independently written renderer or against a locally-run scheduler that does not use the Streamer code.
- 2.3 For the pending application referenced in PATENTS.md ("System, Apparatus, and Method for Generating Avatar", docket 08915-P0003A): could you share the application number or a claims summary, so we can assess whether a simplified single-PC playback loop falls within its scope? We are also aware of US 12,322,020 B1; please indicate whether it is considered relevant to AVTR-1 deployments.
- 2.4 Attachment A-20 (competing products): please confirm that an on-premise avatar embedded in a [ ]-domain product is not considered a competing product or service.
- 2.5 Distribution: installing the software on a customer-owned PC is a third-party distribution under Section 3 of LICENSE-MODEL.md. Do you accept a customer-facing EULA that incorporates Attachment A by reference, or do you require the full license text to be delivered to each customer?

**3. Deployment facts (to size the quote)**

- Number of customer sites in the first 12 months: [ ]
- PCs per site: 1 (NVIDIA RTX [ ], Linux or Windows [ ])
- Avatars per PC: [ ] portraits, prepared offline
- Usage: interactive dialogue with a local LLM/TTS, ~[ ] hours/day
- Network: no external streaming; optional LAN-only remote control

We would appreciate a response within two weeks so we can plan the build. Thank you.

Best regards,
[ ] (name, title)
[ ] (company, address)
[ ] (phone / email)

---

## 한국어 요지 (내부 검토용)

1. **견적 2종 + 1종**: (A) 모델 단독, (B) 모델 + Renderer 상업 라이선스, (C) 참고용 Streamer 상업 라이선스. 가격 구조(배포 건별·PC별·연간·매출 연동)와 포함 범위(업데이트·지원·엔진 빌드)를 요청
2. **범위 확인 5건**
   - 2.1 Renderer 폴더 파일 헤더(Community)와 폴더 라이선스(PolyForm 비상업) 불일치 시 어느 쪽이 우선인지. 특히 모델 호출 코드와 `scripts/build_avtr1_engines.py` 의 상업 사용 가능 여부
   - 2.2 모델·HuBERT 유지 + 자체 렌더러 + 자체 로컬 재생 코드에 대해 특허·라이선스 주장이 없는지 서면 확인
   - 2.3 미공개 출원의 출원번호 또는 청구항 요지 공유 요청. 등록 특허 US 12,322,020 B1 의 관련성 확인
   - 2.4 우리 제품이 "경쟁 제품"(Attachment A-20) 이 아님을 확인
   - 2.5 고객사 PC 설치 시 라이선스 전문 제공 방식(EULA 참조 인용 허용 여부)
3. **배포 규모**: 사이트 수, PC 사양, 아바타 수, 일 사용 시간, 외부 스트리밍 없음

## 회신 후 판단 기준

| 회신 내용 | 판단 |
| --- | --- |
| (B) 비용이 자체 렌더러 개발 공수(`docs/REIMPLEMENTATION_PLAN.md` 추정치)보다 낮고 특허 확인(2.2)이 명확 | Renderer 계약 채택. Streamer 는 로컬 재생 코드로 자체 구성 |
| (B) 비용이 높고 2.1 에서 모델 호출 코드 사용이 허용됨 | 모델 호출 코드 재사용 + 자체 렌더러 구현 |
| 2.1 에서 Renderer 폴더 전체가 비상업으로 확정 | 모델 호출 코드까지 자체 작성. `scripts/build_avtr1_engines.py` 의 I/O 계약을 기준으로 재작성 |
| 2.2 에서 자체 구현에도 특허 주장 | 특허 변호사 FTO 검토 후 결정. 회피 설계 또는 Streamer 계약 |
