-- ===============================================
-- 生产环境特征宽表构建SQL（已添加 bill_no 长度过滤）
-- 训练主表: a_utrm_use_m
-- 生成时间: 2025-12-23
-- 特征数量: 38
-- 过滤条件: LENGTH(bill_no) = 11 (仅保留标准手机号，排除物联网号码)
-- ===============================================

-- 重要说明（为兼容DM字段长度限制）：
-- 1) 本SQL将模型特征列统一改为短列名 f001~f038，避免字段名/字段中文名超长导致写入失败。
-- 2) 预测脚本侧需要把 f001~f038 映射回模型期望的长特征名（可在发布脚本里自动完成）。
-- 3) 映射规则：f001 对应第1个特征、f002 对应第2个特征……顺序与下方 SELECT 列表一致。

-- 使用说明:
-- 1. ${mtaskid} 为当前账期，如 '202512'
-- 2. ${lm_mtaskid} 为上月账期，如 '202511'
-- 3. 在DM数据开发平台执行
-- 4. 此版本添加了 LENGTH(bill_no) = 11 过滤条件，避免物联网号码导致的NULL值

SELECT
    main_agg.bill_no AS bill_no,
    main_agg.city_id AS city_id,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_term_change_cnt AS f001,
    t1_agg.SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_call_fee_365d AS f002,
    t1_agg.AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_grp_pay_rate AS f003,
    t1_agg.COUNT_temp_a_upay_user_attr_m_bak_encrypt_202508_pay_family_flag AS f004,
    t1_agg.COUNT_temp_a_upay_user_attr_m_bak_encrypt_202508_stock_flag AS f005,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mon_gprs_4g AS f006,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_f_use_date AS f007,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_call_cnt AS f008,
    t1_agg.SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_gprs_month_fee AS f009,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mode AS f010,
    t2_agg.AVG_temp_user_base_info_m_bak_encrypt_202508_edu_level_id AS f011,
    t1_agg.MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_roamidd_fee AS f012,
    t2_agg.AVG_temp_user_base_info_m_bak_encrypt_202508_mob_card_cnt AS f013,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_call_dur AS f014,
    t1_agg.SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_unpay_fee AS f015,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_gprs AS f016,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_thir_imei_f_use_date AS f017,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mon_call_cnt AS f018,
    t1_agg.MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_sms_comm_fee AS f019,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_gpu_rate AS f020,
    t2_agg.COUNT_temp_user_base_info_m_bak_encrypt_202508_if_executive AS f021,
    t1_agg.MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_ddd_fee AS f022,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_use_days AS f023,
    t1_agg.COUNT_temp_a_upay_user_attr_m_bak_encrypt_202508_grp_pay_flag_90d AS f024,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_fir_imei_mon_gprs AS f025,
    t1_agg.AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_gprs_comm_fee AS f026,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_sec_imei_gpu_rate AS f027,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_sec_imei_use_days AS f028,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_thir_imei_price AS f029,
    t1_agg.AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_avg_basesup_fee AS f030,
    t1_agg.SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_crinfo_fee AS f031,
    t1_agg.MAX_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_fact_fee AS f032,
    t2_agg.SUM_temp_user_base_info_m_bak_encrypt_202508_oth_card_cnt AS f033,
    t1_agg.SUM_temp_a_upay_user_attr_m_bak_encrypt_202508_call_comm_fee_90d AS f034,
    main_agg.LATEST_temp_a_utrm_use_3m_inline_encrypt_202508_mon_user_gprs_4g AS f035,
    t1_agg.MIN_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_fact_fee AS f036,
    t1_agg.AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_fav_fee AS f037,
    t1_agg.AVG_temp_a_upay_user_attr_m_bak_encrypt_202508_n3m_med_gprs_month_fee AS f038

FROM
    (
    -- 主表：终端使用月表（仅11位标准手机号）
    SELECT
        bill_no AS bill_no,
        MAX(city_id) AS city_id,
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
        AND LENGTH(bill_no) = 11  -- 仅保留11位标准手机号
        AND SUBSTR(bill_no, 1, 1) = '1'  -- 排除字母开头的宽带账号
    GROUP BY
        bill_no
    ) AS main_agg

LEFT JOIN (
    -- 副表1：用户属性月表（仅11位标准手机号）
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
        AND LENGTH(bill_no) = 11  -- 仅保留11位标准手机号
        AND SUBSTR(bill_no, 1, 1) = '1'  -- 排除字母开头的宽带账号
    GROUP BY
        bill_no
) AS t1_agg
    ON main_agg.bill_no = t1_agg.bill_no

LEFT JOIN (
    -- 副表2：用户基础信息月表（仅11位标准手机号）
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
        AND LENGTH(bill_no) = 11  -- 仅保留11位标准手机号
        AND SUBSTR(bill_no, 1, 1) = '1'  -- 排除字母开头的宽带账号
    GROUP BY
        bill_no
) AS t2_agg
    ON main_agg.bill_no = t2_agg.bill_no
;
