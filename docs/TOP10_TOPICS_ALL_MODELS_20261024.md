# Mười topic của từng pipeline — 24/10/2026

Xuất trực tiếp từ sáu catalog đang có trong ứng dụng; không train/refit.
ID topic chỉ có ý nghĩa bên trong từng pipeline, không phải phép ghép cặp giữa model.
Nhãn là diễn giải từ keywords và tài liệu đại diện, chưa có chuyên gia xác nhận.
BatteryBERT full-document · trained là topic chính hiện tại. Dữ liệu huấn luyện có Title–Abstract đầy đủ, chưa có toàn văn bài báo.
LDA dưới đây là bản đối chứng K10; bản chọn K của Stage2 lịch sử là một thí nghiệm riêng.

## BatteryBERT full-document · trained

| ID | Diễn giải trong catalog | Tài liệu | Top10 keywords |
|---|---|---:|---|
| 0 | Oxide Ti/Sn, titanate và composite điện cực | 1,408 | TiO2, titanium, titanate, solution, dioxide, SnO2, oxide, anode, carbon, composite |
| 1 | Prussian blue và các hợp chất điện cực chứa sắt | 1,483 | iron, prussian_blue, solution, sodium, ferrous, anode, prussian, cathode, ferrocyanide, FeS2 |
| 2 | Oxide lớp P2/O3 và biến đổi cấu trúc điện cực | 1,717 | P2, layer, oxide, cathode, O3, layered_oxide, type, transition, phase, metal |
| 3 | Điện ly, polymer, muối và phụ gia giao diện | 1,426 | electrolyte, polymer, additive, organic, solvent, salt, carbonate, sodium, electrode, non-aqueous |
| 4 | Phosphate vanadium, NVP/NASICON và composite carbon | 1,192 | phosphate, Na3V2(PO4), vanadium, nvp, phosphorus, carbon, sodium, cathode, composite, fluorophosphate |
| 5 | Anode hợp kim Sn/Sb, graphene và MXene | 1,266 | graphene, antimony, mxene, sb, anode, composite, sn, alloy, electrode, tin |
| 6 | Phần cứng cell và thiết bị chế tạo, kiểm tra pin | 925 | connect, plate, device, end, pole, piece, body, electric, electrode, collector |
| 7 | Cấu trúc pha, trật tự vacancy và dẫn ion | 1,035 | phase, na, NaxCoO2, sodium, compound, electrode, cathode, ion, solid_electrolyte, active |
| 8 | Hard carbon, carbon xốp và tiền chất sinh khối | 1,653 | hard_carbon, carbon, biomass, porous, carbonize, precursor, carbonization, dry, asphalt, anode |
| 9 | Sulfide/selenide, MoS2 và composite carbon | 1,597 | MoS2, sulfide, disulfide, molybdenum, carbon, composite, selenide, anode, solution, SnS2 |

Ngoài 10 topic: 0 tài liệu outlier (ID −1).

## BatteryBERT fine-tuned · optimized

| ID | Diễn giải trong catalog | Tài liệu | Top10 keywords |
|---|---|---:|---|
| 0 | Anode MoS2, dichalcogenide lớp và MXene | 1,031 | MoS2, graphene, mxene, molybdenum, disulfide, nanosheets, SnS2, MoSe2, anode, composite |
| 1 | Cathode phosphate vanadium và NASICON | 1,342 | phosphate, vanadium, Na3V2(PO4), nvp, sodium, cathode, carbon, PO4, F-, fluorophosphate |
| 2 | Cathode Prussian blue và hợp chất sắt | 1,353 | prussian_blue, iron, sodium, solution, prussian, cathode, ferrocyanide, ferrous, anode, iron-based |
| 3 | Cấu trúc cell, bộ góp dòng và lắp ráp pin | 1,084 | connect, plate, device, end, electrode, collector, pole, electric, positive, piece |
| 4 | Điện ly, muối, dung môi và polymer | 1,532 | electrolyte, polymer, additive, organic, salt, solvent, sodium, carbonate, electrode, solid_electrolyte |
| 5 | Anode sulfide/selenide và composite carbon | 1,235 | sulfide, carbon, composite, selenide, CoSe2, cobalt, FeS2, anode, solution, nickel |
| 6 | Anode hợp kim Sn/Sb, phosphorus và chuyển hóa | 915 | antimony, sb, tin, sn, anode, alloy, composite, carbon, phosphorus, Sb2S3 |
| 7 | Hard carbon, carbon xốp và tiền chất sinh khối | 1,703 | hard_carbon, carbon, biomass, porous, carbonize, precursor, carbonization, dry, anode, asphalt |
| 8 | Anode oxide Ti/Sn và titanate | 1,444 | TiO2, titanium, titanate, dioxide, SnO2, solution, oxide, anode, sodium, composite |
| 9 | Cathode oxide lớp P2/O3 và kim loại chuyển tiếp | 2,063 | P2, layer, oxide, cathode, O3, layered_oxide, phase, transition, type, metal |

Ngoài 10 topic: 0 tài liệu outlier (ID −1).

## BatteryBERT KMeans (pretrained)

| ID | Diễn giải trong catalog | Tài liệu | Top10 keywords |
|---|---|---:|---|
| 0 | Wet-chemical synthesis of electrode materials | 2,309 | solution, mix, dry, water, stir, hour, source, method, precursor, wash |
| 1 | Active-material formulations and electrode coatings | 1,677 | electrode, active, active material, negative, battery, negative electrode, layer, positive, material, sodium_ion |
| 2 | Mixed electrode performance and electrolyte studies | 1,624 | electrolyte, capacity, cycle, electrochemical, high, na-ion, performance, na, anode, result |
| 3 | Anode composites and cycling performance | 1,622 | ma, capacity, mah, cycle, high, sib, capacity mah, rate, performance, anode |
| 4 | Sulfide/selenide-carbon composite anodes | 1,603 | MoS2, storage, cycle, sib, nanosheets, capacity, anode, sodium storage, TiO2, high |
| 5 | Inorganic electrode compositions and chemical claims | 1,470 | sodium, formula, material, metal, oxide, anode material, compound, anode, sodium_ion, sodium_ion battery |
| 6 | Layered-oxide and phosphate cathode materials | 1,351 | P2, cathode, capacity, Na3V2(PO4), type, cycle, cathode material, O3, electrochemical, P2 type |
| 7 | Crystal structure, redox and intercalation studies | 1,291 | na, phase, redox, structural, na-ion, property, Na+, structure, x-ray, diffusion |
| 8 | Cell hardware and assembly equipment | 480 | connect, plate, device, end, box, rod, cover, fix, electric, groove |
| 9 | Reviews and research perspectives | 275 | sib, application, energy_storage, advance, design, strategy, mechanism, focus, perspective, cost |

Ngoài 10 topic: 0 tài liệu outlier (ID −1).

## LDA

| ID | Diễn giải trong catalog | Tài liệu | Top10 keywords |
|---|---|---:|---|
| 0 | Sodium-storage anodes and carbon composites | 3,486 | anode, cycle, capacity, carbon, high, sodium_ion, performance, storage, material, sib |
| 1 | Non-aqueous electrolytes and ionic transport | 997 | battery, electrolyte, study, sodium_ion, deliver, sodium, use, conductivity, ionic, analysis |
| 2 | Cathode electrochemistry and vanadium phosphates | 3,559 | capacity, cathode, cycle, electrochemical, material, na, performance, Na+, high, na-ion |
| 3 | Electrode-material inventions (mixed materials) | 869 | material, anode, sodium_ion, battery, sodium, cathode, advance, use, performance, high |
| 4 | Transition-metal and Prussian-white formulations | 78 | nc, material, iron, prussian_blue, manganese, coat, layered_oxide, oxide, nickel, Ni1 |
| 5 | Layered oxide cathodes and P2/O3 structures | 84 | layer, transition, P2, phase, metal, cathode, oxide, type, O3, material |
| 6 | Active electrode materials and characterization | 1,286 | electrode, battery, material, use, sodium_ion, positive, layer, spectroscopy, active, negative |
| 7 | Material synthesis and preparation processes | 2,529 | solution, obtain, material, prepare, development, mix, add, battery, sodium_ion, water |
| 8 | Battery devices, housings and assembly | 331 | successfully, battery, connect, device, plate, sodium_ion, shell, provide, core, end |
| 9 | Hard-carbon preparation and related materials | 483 | carbon, material, hard_carbon, precursor, mxene, obtain, prepare, method, use, temperature |

Ngoài 10 topic: 0 tài liệu outlier (ID −1).

## BERTopic–BGE

| ID | Diễn giải trong catalog | Tài liệu | Top10 keywords |
|---|---|---:|---|
| 0 | Electrode-material preparation patents | 3,552 | obtain, material, prepare, solution, mix, source, use, add, precursor, dry |
| 1 | Layered-oxide and polyanion cathodes | 2,650 | P2, cathode, capacity, cycle, electrochemical, type, na, phase, O3, Na3V2(PO4) |
| 2 | Alloy/conversion and carbon-based anodes | 1,261 | sn, capacity, SnO2, ma, SnS2, anode, sb, cycle, storage, mah |
| 3 | Sulfide/selenide and carbon-composite anodes | 1,008 | MoS2, FeS2, cycle, storage, sib, CoSe2, MoSe2, capacity, ma, mah |
| 4 | Prussian-blue analogue cathodes | 499 | prussian_blue, ferrocyanide, prussian, hexacyanoferrate, analogue, white, Fe(CN), water, solution, pb |
| 5 | Titanate and TiO2 anodes | 425 | TiO2, Na2Ti3O7, anatase, nto, ma, cycle, titanate, storage, capacity, anode |
| 6 | Electrolyte salts, solvents and additives | 397 | electrolyte, additive, solvent, salt, non-aqueous, solid_electrolyte, organic, polymer, sodium, carbonate |
| 7 | Battery hardware and manufacturing devices | 382 | connect, plate, device, end, fixedly, box, rod, fix, cover, frame |
| 8 | Sodium cobalt oxides and superconductivity | 219 | NaxCoO2, NaCrO2, cuo, Co3O4, CoO2, superconducting, yh, superconductivity, phase, magnetic |
| 9 | MXene-based sodium-storage electrodes | 189 | mxene, Ti3C2Tx, mxenes, Ti3C2, carbide, mxene-based, two-dimensional, titanium, etch, TiS2 |

Ngoài 10 topic: 3,120 tài liệu outlier (ID −1).

## BERTopic–BatteryBERT (pretrained)

| ID | Diễn giải trong catalog | Tài liệu | Top10 keywords |
|---|---|---:|---|
| 0 | Electrode performance (broad Publication cluster) | 7,261 | capacity, cycle, ma, mah, high, performance, electrochemical, sib, rate, storage |
| 1 | Material preparation methods (broad Patent cluster) | 5,449 | obtain, solution, prepare, material, mix, use, add, sodium, method, source |
| 2 | Reviews and research outlook | 218 | review, recent, summarize, progress, research, discuss, future, challenge, development, sib |
| 3 | 2D anodes and first-principles calculations | 164 | na, monolayer, adsorption, li, ev, barrier, atom, phosphorene, calculation, first-principles |
| 4 | Fixtures, protection and inspection equipment | 156 | connect, plate, fixedly, utility, rod, model, end, box, groove, fix |
| 5 | Assembly, heating and mechanical equipment | 101 | device, connect, plate, fix, frame, rotate, roller, rod, box, end |
| 6 | Battery packs, management and control | 80 | module, unit, voltage, pack, data, battery, charge, power, management, detect |
| 7 | Sodium cobaltate superconductivity | 79 | NaxCoO2, superconducting, yh, CoO2, superconductivity, magnetic, t-c, center, dot, superconductor |
| 8 | Cell housing and terminals | 48 | portion, terminal, electric, monomer, assembly, toy, second, cover, end, direction |
| 9 | Mixed high-performance electrode materials | 37 | high-performance, snsb-porous, vapor-redistribution, Se4P4, Na2C12H6O4, ultracycle-stable, photoassisted, krohnkite-type, na-ion, superior |

Ngoài 10 topic: 109 tài liệu outlier (ID −1).
