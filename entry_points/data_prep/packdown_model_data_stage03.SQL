CREATE OR REPLACE TABLE `{staging_project}.{staging_dataset}.PACKDOWN_MODEL_DATA_STAGE03`
PARTITION BY CAL_DT
OPTIONS(expiration_timestamp=TIMESTAMP(DATE_ADD(CURRENT_DATE, INTERVAL 1 DAY)))
AS(

SELECT
  A. CAL_DT
, B. SUB_DEPT_NBR
, B. EXT_CLASS_NBR
, B. EXT_SUB_CLASS_NBR
, A. SKU_STR_ID
, B. STR_X
, B. STR_Y
, B. STR_Z
, COALESCE(MAX(CASE WHEN D. FSCL_WK_BGN_DT BETWEEN DATE_SUB(A. CAL_DT, INTERVAL 6 DAY) AND A. CAL_DT THEN EFF_FCST_QTY END),-1) AS PREV_FCST_WK0
, COALESCE(MAX(CASE WHEN D. FSCL_WK_BGN_DT BETWEEN DATE_SUB(A. CAL_DT, INTERVAL 14 DAY) AND DATE_SUB(A. CAL_DT, INTERVAL 7 DAY) THEN EFF_FCST_QTY END),-1)  AS PREV_FCST_WK1
, COALESCE(MAX(CASE WHEN D. FSCL_WK_BGN_DT BETWEEN DATE_SUB(A. CAL_DT, INTERVAL 21 DAY) AND DATE_SUB(A. CAL_DT, INTERVAL 15 DAY) THEN EFF_FCST_QTY END),-1) AS PREV_FCST_WK2
, COALESCE(MAX(CASE WHEN D. FSCL_WK_BGN_DT BETWEEN DATE_SUB(A. CAL_DT, INTERVAL 28 DAY) AND DATE_SUB(A. CAL_DT, INTERVAL 22 DAY) THEN EFF_FCST_QTY END),-1) AS PREV_FCST_WK3
, COALESCE(MAX(CASE WHEN D. FSCL_WK_BGN_DT BETWEEN DATE_SUB(A. CAL_DT, INTERVAL 35 DAY) AND DATE_SUB(A. CAL_DT, INTERVAL 29 DAY) THEN EFF_FCST_QTY END),-1) AS PREV_FCST_WK4
, COALESCE(MAX(CASE WHEN D. FSCL_WK_BGN_DT BETWEEN DATE_SUB(A. CAL_DT, INTERVAL 42 DAY) AND DATE_SUB(A. CAL_DT, INTERVAL 36 DAY) THEN EFF_FCST_QTY END),-1) AS PREV_FCST_WK5
, COALESCE(MAX(CASE WHEN D. FSCL_WK_BGN_DT BETWEEN DATE_SUB(A. CAL_DT, INTERVAL 49 DAY) AND DATE_SUB(A. CAL_DT, INTERVAL 43 DAY) THEN EFF_FCST_QTY END),-1) AS PREV_FCST_WK6

, ROUND(COALESCE(AVG(RTN_UNTS),0),4) AVG_RTN_UNTS
, ROUND(COALESCE(AVG(R52_RTN_UNTS),0),4) AVG_R52_RTN_UNTS
FROM `{staging_project}.{staging_dataset}.PACKDOWN_MODEL_DATA_STAGE01` A
INNER JOIN `{staging_project}.{staging_dataset}.THD_SKU_ACTIVE` B
  ON A. SKU_STR_ID = B. SKU_STR_ID

INNER JOIN `{edw_project}.{edw_dataset_shared}.CAL_PRD_HIER_FD` C
  ON A. CAL_DT = C. CAL_DT

LEFT JOIN  `{edw_project}.{edw_dataset_schn_sls_fcst}.WKLY_SLS_FCST_HIST` d
    ON A. STR_NBR = D. LOC_NBR
    and A. SKU_NBR = D. SKU_NBR
    AND D. FSCL_WK_BGN_DT BETWEEN DATE_SUB(A. CAL_DT, INTERVAL 14 week) and A. CAL_DT
    and D. AS_OF_WK = DATE_SUB(D. FSCL_WK_BGN_DT, INTERVAL 7 DAY)

LEFT JOIN (
           SELECT
            A. SLS_DT
          , A. SKU_STR_ID
          , A. GROSS_SLS_AMT SLS_AMT
          , ABS(GROSS_RTN_UNTS) RTN_UNTS
          , SUM(ABS(GROSS_RTN_UNTS)) OVER(PARTITION BY A. SKU_STR_ID ORDER BY UNIX_DATE(A. SLS_DT) RANGE BETWEEN 364 PRECEDING AND CURRENT ROW) R52_RTN_UNTS
          FROM `{staging_project}.{staging_dataset}.SKU_STR_SALES_HIST` A
          INNER JOIN `{staging_project}.{staging_dataset}.PACKDOWN_MODEL_DATA_STAGE01` B
            ON A. SKU_STR_ID = B. SKU_STR_ID
          WHERE SLS_DT > DATE_SUB(CURRENT_DATE, INTERVAL 104 WEEK)

        ) E
    ON E. SLS_DT BETWEEN DATE_SUB(A. CAL_DT, INTERVAL 52 WEEK) AND A. CAL_DT
    AND A. SKU_STR_ID = E. SKU_STR_ID

GROUP BY
      CAL_DT
     , SUB_DEPT_NBR
     , EXT_CLASS_NBR
     , EXT_SUB_CLASS_NBR
     , SKU_STR_ID
     , STR_X
     , STR_Y
     , STR_Z
     , SKU_STR_ID


);