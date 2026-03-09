-- ===============================================
-- 生产环境特征宽表构建SQL（已替换实际表名）
-- 训练主表: a_utrm_use_m
-- 生成时间: 2025-12-23
-- 特征数量: 38
-- ===============================================

-- 使用说明:
-- 1. ${mtaskid} 为当前账期，如 '202512'
-- 2. ${lm_mtaskid} 为上月账期，如 '202511'
-- 3. 在DM数据开发平台执行

SELECT
    main_agg.bill_no AS bill_no,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_term_change_cnt AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_term_change_cnt,
    t1_agg.SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_call_fee_365d AS SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_call_fee_365d,
    t1_agg.AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_grp_pay_rate AS AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_grp_pay_rate,
    t1_agg.COUNT_temp_a_upay_user_attr_m_bak_encrypt_202508_pay_family_flag AS COUNT_temp_a_upay_user_attr_m_bak_encrypt_202508_pay_family_flag,
    t1_agg.COUNT_temp_a_upay_user_attr_m_bak_encrypt_202508_stock_flag AS COUNT_temp_a_upay_user_attr_m_bak_encrypt_202508_stock_flag,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mon_gprs_4g AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mon_gprs_4g,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_f_use_date AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_f_use_date,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_call_cnt AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_call_cnt,
    t1_agg.SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_gprs_month_fee AS SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_gprs_month_fee,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mode AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mode,
    t2_agg.AVG_temp_user_base_info_m_bak_encrypt_202508_edu_level_id AS AVG_temp_user_base_info_m_bak_encrypt_202508_edu_level_id,
    t1_agg.MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_roamidd_fee AS MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_roamidd_fee,
    t2_agg.AVG_temp_user_base_info_m_bak_encrypt_202508_mob_card_cnt AS AVG_temp_user_base_info_m_bak_encrypt_202508_mob_card_cnt,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_call_dur AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_call_dur,
    t1_agg.SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_unpay_fee AS SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_unpay_fee,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_gprs AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_gprs,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_thir_imei_f_use_date AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_thir_imei_f_use_date,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mon_call_cnt AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mon_call_cnt,
    t1_agg.MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_sms_comm_fee AS MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_sms_comm_fee,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_gpu_rate AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_gpu_rate,
    t2_agg.COUNT_temp_user_base_info_m_bak_encrypt_202508_if_executive AS COUNT_temp_user_base_info_m_bak_encrypt_202508_if_executive,
    t1_agg.MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_ddd_fee AS MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_ddd_fee,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_use_days AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_use_days,
    t1_agg.COUNT_temp_a_upay_user_attr_m_bak_encrypt_202508_grp_pay_flag_90d AS COUNT_temp_a_upay_user_attr_m_bak_encrypt_202508_grp_pay_flag_90d,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mon_gprs AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mon_gprs,
    t1_agg.AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_gprs_comm_fee AS AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_gprs_comm_fee,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_sec_imei_gpu_rate AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_sec_imei_gpu_rate,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_sec_imei_use_days AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_sec_imei_use_days,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_thir_imei_price AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_thir_imei_price,
    t1_agg.AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_avg_basesup_fee AS AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_avg_basesup_fee,
    t1_agg.SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_crinfo_fee AS SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_crinfo_fee,
    t1_agg.MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_fact_fee AS MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_fact_fee,
    t2_agg.SUM_temp_user_base_info_m_bak_encrypt_202508_oth_card_cnt AS SUM_temp_user_base_info_m_bak_encrypt_202508_oth_card_cnt,
    t1_agg.SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_call_comm_fee_90d AS SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_call_comm_fee_90d,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_gprs_4g AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_gprs_4g,
    t1_agg.MIN_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_fact_fee AS MIN_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_fact_fee,
    t1_agg.AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_fav_fee AS AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_fav_fee,
    t1_agg.AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_med_gprs_month_fee AS AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_med_gprs_month_fee

FROM
    (
    -- 主表：终端使用月表
    SELECT
        bill_no AS bill_no,
        MAX(term_change_cnt) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_term_change_cnt,
        MAX(fir_imei_mon_gprs_4g) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mon_gprs_4g,
        MAX(fir_imei_f_use_date) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_f_use_date,
        MAX(mon_user_call_cnt) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_call_cnt,
        MAX(fir_imei_mode) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mode,
        MAX(mon_user_call_dur) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_call_dur,
        MAX(mon_user_gprs) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_gprs,
        MAX(thir_imei_f_use_date) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_thir_imei_f_use_date,
        MAX(fir_imei_mon_call_cnt) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mon_call_cnt,
        MAX(fir_imei_gpu_rate) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_gpu_rate,
        MAX(fir_imei_use_days) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_use_days,
        MAX(fir_imei_mon_gprs) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mon_gprs,
        MAX(sec_imei_gpu_rate) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_sec_imei_gpu_rate,
        MAX(sec_imei_use_days) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_sec_imei_use_days,
        MAX(thir_imei_price) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_thir_imei_price,
        MAX(mon_user_gprs_4g) AS LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_gprs_4g
    FROM
        hlwyx_ns3_hive_db.a_utrm_use_m
    WHERE
        p_mon = '${mtaskid}'
    GROUP BY
        bill_no
    ) AS main_agg

LEFT JOIN (
    -- 副表1：用户属性月表
    SELECT
        bill_no AS bill_no,
        SUM(n3m_call_fee) AS SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_call_fee_365d,
        AVG(grp_pay_rate) AS AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_grp_pay_rate,
        COUNT(pay_family_flag) AS COUNT_temp_a_upay_user_attr_m_bak_encrypt_202508_pay_family_flag,
        COUNT(stock_flag) AS COUNT_temp_a_upay_user_attr_m_bak_encrypt_202508_stock_flag,
        SUM(n3m_gprs_month_fee) AS SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_gprs_month_fee,
        MAX(roamidd_fee) AS MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_roamidd_fee,
        SUM(unpay_fee) AS SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_unpay_fee,
        MAX(sms_comm_fee) AS MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_sms_comm_fee,
        MAX(ddd_fee) AS MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_ddd_fee,
        COUNT(grp_pay_flag) AS COUNT_temp_a_upay_user_attr_m_bak_encrypt_202508_grp_pay_flag_90d,
        AVG(gprs_comm_fee) AS AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_gprs_comm_fee,
        AVG(n3m_avg_basesup_fee) AS AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_avg_basesup_fee,
        SUM(n3m_crinfo_fee) AS SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_crinfo_fee,
        MAX(n3m_fact_fee) AS MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_fact_fee,
        SUM(call_comm_fee) AS SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_call_comm_fee_90d,
        MIN(n3m_fact_fee) AS MIN_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_fact_fee,
        AVG(n3m_fav_fee) AS AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_fav_fee,
        AVG(n3m_med_gprs_month_fee) AS AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_med_gprs_month_fee
    FROM
        hlwyx_ns3_hive_db.a_upay_user_attr_m
    WHERE
        p_mon = '${mtaskid}'
    GROUP BY
        bill_no
) AS t1_agg
    ON main_agg.bill_no = t1_agg.bill_no

LEFT JOIN (
    -- 副表2：用户基础信息月表
    SELECT
        bill_no AS bill_no,
        AVG(edu_level_id) AS AVG_temp_user_base_info_m_bak_encrypt_202508_edu_level_id,
        AVG(mob_card_cnt) AS AVG_temp_user_base_info_m_bak_encrypt_202508_mob_card_cnt,
        COUNT(if_executive) AS COUNT_temp_user_base_info_m_bak_encrypt_202508_if_executive,
        SUM(oth_card_cnt) AS SUM_temp_user_base_info_m_bak_encrypt_202508_oth_card_cnt
    FROM
        hlwyx_ns3_hive_db.d_bdapp_l_attr_user_base_info_m
    WHERE
        p_mon = '${mtaskid}'
    GROUP BY
        bill_no
) AS t2_agg
    ON main_agg.bill_no = t2_agg.bill_no
;
