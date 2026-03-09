-- ===============================================
-- 生产环境特征宽表构建SQL
-- 训练主表: temp_a_utrm_use_3m_inline_encrypt_202511
-- 生成时间: 2026-03-04 18:09:01
-- 特征数量: 26
-- ===============================================

-- 使用说明:
-- 1. 将 {{TARGET_TABLE}} 替换为你的目标宽表名
-- 2. 将 {{MainTable_xxx}} 替换为实际生产主表名
-- 3. 将 {{Table_xxx}} 替换为实际生产副表名
-- 4. 确认JOIN字段在生产表中存在
-- 5. 在DM数据开发平台执行

DROP TABLE IF EXISTS {{TARGET_TABLE}};
CREATE TABLE {{TARGET_TABLE}} AS
SELECT
    main_agg.bill_no AS bill_no,
    t2_agg.LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_l1m_pay_fee AS f001,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_sec_imei_l_use_date AS f002,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_mon_gprs AS f003,
    t2_agg.LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_n3m_fact_fee AS f004,
    t1_agg.LATEST_temp_user_base_info_m_bak_encrypt_202511_occu AS f005,
    t1_agg.LATEST_temp_user_base_info_m_bak_encrypt_202511_sub_occu_name AS f006,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_thir_imei_use_days AS f007,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_sec_imei_use_days AS f008,
    t2_agg.LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_sc_freeze AS f009,
    t1_agg.LATEST_temp_user_base_info_m_bak_encrypt_202511_community_price AS f010,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_f_use_date AS f011,
    t2_agg.LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_sms_comm_fee AS f012,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_gpu_rate AS f013,
    t1_agg.LATEST_temp_user_base_info_m_bak_encrypt_202511_income_level AS f014,
    t2_agg.LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_l2m_pay_fee AS f015,
    t2_agg.LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_sc_used AS f016,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_price AS f017,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_mdl AS f018,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_mon_gprs_4g AS f019,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_mode AS f020,
    t2_agg.LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_book_bj_bal AS f021,
    t2_agg.LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_book_kzj_bal AS f022,
    t2_agg.LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_l3m_call_fee AS f023,
    t2_agg.LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_woff_bj_fee AS f024,
    t2_agg.LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_month_fee AS f025,
    t1_agg.LATEST_temp_user_base_info_m_bak_encrypt_202511_car_probabi AS f026

FROM
    (
    SELECT
        bill_no AS bill_no,
        MAX(sec_imei_l_use_date) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_sec_imei_l_use_date,
        MAX(fir_imei_mon_gprs) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_mon_gprs,
        MAX(thir_imei_use_days) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_thir_imei_use_days,
        MAX(sec_imei_use_days) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_sec_imei_use_days,
        MAX(fir_imei_f_use_date) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_f_use_date,
        MAX(fir_imei_gpu_rate) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_gpu_rate,
        MAX(fir_imei_price) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_price,
        MAX(fir_imei_mdl) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_mdl,
        MAX(fir_imei_mon_gprs_4g) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_mon_gprs_4g,
        MAX(fir_imei_mode) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202511_fir_imei_mode
    FROM
        {{MainTable_temp_a_utrm_use_3m_inline_encrypt_202511}}
    GROUP BY
        bill_no
    ) AS main_agg
LEFT JOIN (
    SELECT
        bill_no AS bill_no,
        MAX(occu) AS LATEST_temp_user_base_info_m_bak_encrypt_202511_occu,
        MAX(sub_occu_name) AS LATEST_temp_user_base_info_m_bak_encrypt_202511_sub_occu_name,
        MAX(community_price) AS LATEST_temp_user_base_info_m_bak_encrypt_202511_community_price,
        MAX(income_level) AS LATEST_temp_user_base_info_m_bak_encrypt_202511_income_level,
        MAX(car_probabi) AS LATEST_temp_user_base_info_m_bak_encrypt_202511_car_probabi
    FROM
        {{Table_temp_user_base_info_m_bak_encrypt_202511}}
    GROUP BY
        bill_no
) AS t1_agg
    ON main_agg.bill_no = t1_agg.bill_no

LEFT JOIN (
    SELECT
        bill_no AS bill_no,
        MAX(l1m_pay_fee) AS LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_l1m_pay_fee,
        MAX(n3m_fact_fee) AS LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_n3m_fact_fee,
        MAX(sc_freeze) AS LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_sc_freeze,
        MAX(sms_comm_fee) AS LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_sms_comm_fee,
        MAX(l2m_pay_fee) AS LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_l2m_pay_fee,
        MAX(sc_used) AS LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_sc_used,
        MAX(book_bj_bal) AS LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_book_bj_bal,
        MAX(book_kzj_bal) AS LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_book_kzj_bal,
        MAX(l3m_call_fee) AS LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_l3m_call_fee,
        MAX(woff_bj_fee) AS LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_woff_bj_fee,
        MAX(month_fee) AS LATEST_temp_a_upay_user_attr_m_bak_encrypt_202511_month_fee
    FROM
        {{Table_temp_a_upay_user_attr_m_bak_encrypt_202511}}
    GROUP BY
        bill_no
) AS t2_agg
    ON main_agg.bill_no = t2_agg.bill_no
;
