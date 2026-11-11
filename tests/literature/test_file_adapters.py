from __future__ import annotations

from uhpc_ml_app.literature.adapters.cnki_refworks import CnkiRefWorksAdapter
from uhpc_ml_app.literature.adapters.scopus_csv import ScopusCsvAdapter
from uhpc_ml_app.literature.adapters.wos_ris import WosRisAdapter


def test_wos_ris_adapter_parses_single_record() -> None:
    ris = (
        "TY  - JOUR\n"
        "TI  - Bond behaviour of UHPC-NC interface with Sa = 3.2 μm\n"
        "AU  - Smith, John\n"
        "AU  - Wang, Li\n"
        "PY  - 2022\n"
        "JO  - Cement and Concrete Research\n"
        "DO  - 10.1016/j.cemconres.2022.000001\n"
        "AB  - The interface roughness Sa = 3.2 was measured. "
        "Tensile bond strength reached 4.1 MPa.\n"
        "ER  -\n"
    )
    records = list(WosRisAdapter().ingest(ris))
    assert len(records) == 1
    rec = records[0]
    assert rec.source == "wos"
    assert rec.doi == "10.1016/j.cemconres.2022.000001"
    assert rec.first_author == "Smith"
    assert rec.year == 2022
    assert rec.roughness_sa == 3.2
    assert rec.tensile_bond_strength == 4.1


def test_scopus_csv_adapter_parses_two_rows() -> None:
    csv_text = (
        "Authors,Title,Year,Source title,DOI,Abstract\n"
        '"Zhang, M.; Chen, Q.","Slant shear of UHPC-NC",2021,"Construction and Building Materials",'
        '10.1016/j.conbuildmat.2021.000002,"Sa = 5.0 and tensile bond strength was 3.9 MPa."\n'
        '"Wang, L.","Roughness effect on repair bond",2020,"J Building Eng",'
        '10.1016/j.jobe.2020.000003,"The fiber volume fraction was 2.5 and w/b = 0.2."\n'
    )
    records = list(ScopusCsvAdapter().ingest(csv_text))
    assert len(records) == 2
    first = records[0]
    assert first.source == "scopus"
    assert first.doi == "10.1016/j.conbuildmat.2021.000002"
    assert first.year == 2021
    assert first.roughness_sa == 5.0
    assert first.tensile_bond_strength == 3.9


def test_cnki_refworks_adapter_parses_refworks_block() -> None:
    text = (
        "RT Journal Article\n"
        "A1 张伟\n"
        "T1 UHPC修复界面粘结性能研究\n"
        "YR 2022\n"
        "JF 混凝土\n"
        "DO 10.1234/cnki-sample\n"
        "AB 测试结果表明 Sa = 4.5, 抗剪粘结强度 shear bond strength = 10.2 MPa.\n"
        "\n"
        "RT Thesis\n"
        "A1 李明\n"
        "T1 界面处理工艺对 UHPC-NC 界面粘结性能的影响\n"
        "YR 2021\n"
        "AB w/b = 0.22, fiber volume fraction = 2.0.\n"
    )
    records = list(CnkiRefWorksAdapter().ingest(text))
    assert len(records) == 2
    assert records[0].source == "cnki"
    assert records[0].doi == "10.1234/cnki-sample"
    assert records[0].first_author == "张伟"
    assert records[0].roughness_sa == 4.5
    assert records[0].shear_bond_strength == 10.2
    assert records[1].water_binder_ratio == 0.22


def test_cnki_refworks_adapter_also_accepts_ris() -> None:
    ris = "TY  - JOUR\nTI  - 中文论文 RIS 导出\nA1  - 王芳\nPY  - 2023\nAB  - Sa = 6.0 μm\nER  -\n"
    records = list(CnkiRefWorksAdapter().ingest(ris))
    assert len(records) == 1
    assert records[0].roughness_sa == 6.0
