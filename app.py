# นำเข้าไลบรารีที่ใช้ในแอป
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ทำนายการซื้อสินค้า",
    layout="centered"
)

# โหลดโมเดลและค่า threshold ที่บันทึกไว้ จากโฟลเดอร์เดียวกับ app.py
# และ cache ไว้ จะได้ไม่โหลดใหม่ทุกครั้งที่กดปุ่ม
@st.cache_resource
def load_bundle():
    return joblib.load(Path(__file__).parent / "purchase_intent_pipeline.joblib")

bundle = load_bundle()
pipeline = bundle["pipeline"]
threshold = bundle["threshold"]

# ชื่อกิจกรรมที่แสดงในหน้าเว็บ
activities = {"view": "ดูสินค้า", "cart": "เพิ่มลงตะกร้า"}


# ทำนายจากข้อมูลของ 5 กิจกรรมแรก
def predict(features):
    data = pd.DataFrame([features])
    score = pipeline.predict_proba(data)[0, 1]
    prediction = int(score > threshold)

    return prediction, score

# อธิบายว่าข้อมูลแต่ละช่องเพิ่มหรือลดคะแนนอย่างไร
def explain(features):
    prep = pipeline.named_steps["columntransformer"]
    lr = pipeline.named_steps["logisticregression"]
    scaler = prep.named_transformers_["standardscaler"]

    # เปรียบเทียบกับค่าเฉลี่ยจากข้อมูลฝึกและกิจกรรมดูสินค้า
    reference = dict(zip(bundle["num_cols"], scaler.mean_))
    reference["first_event_type"] = "view"
    reference["fifth_event_type"] = "view"

    data = pd.DataFrame([features, reference])
    ready = prep.transform(data)
    effects = (ready[0] - ready[1]) * lr.coef_[0]
    names = prep.get_feature_names_out()

    labels = {
        "cart_count_first5": f"เพิ่มลงตะกร้า {features['cart_count_first5']} ครั้ง",
        "unique_products_first5": f"สินค้าไม่ซ้ำกัน {features['unique_products_first5']} รายการ",
        "unique_categories_first5": f"สินค้าอยู่ใน {features['unique_categories_first5']} หมวดหมู่",
        "time_span_first5": f"เวลาจากกิจกรรมแรกถึงกิจกรรมที่ 5: {features['time_span_first5'] / 60:g} นาที",
        "mean_price_first5": f"ราคาเฉลี่ย {features['mean_price_first5']:g}",
        "max_price_first5": f"ราคาสูงสุด {features['max_price_first5']:g}",
        "first_event_type": f"กิจกรรมแรก: {activities[features['first_event_type']]}",
        "fifth_event_type": f"กิจกรรมที่ 5: {activities[features['fifth_event_type']]}"
    }

    # รวมผลของแต่ละช่อง โดยรวม view/cart ให้เป็นกิจกรรมช่องเดียว
    results = []
    for col in bundle["feature_cols"]:
        total = 0
        numeric_name = "standardscaler__" + col
        category_prefix = "onehotencoder__" + col + "_"

        for name, effect in zip(names, effects):
            if name == numeric_name or name.startswith(category_prefix):
                total += effect

        results.append({"Factor": labels[col], "Effect": total})

    return pd.DataFrame(results)

# แสดงฟอร์มสรุปข้อมูล 5 กิจกรรมแรก
st.title("ทำนายการซื้อสินค้า")
st.caption("กรอกข้อมูลจาก 5 กิจกรรมแรกในการเข้าเว็บครั้งเดียวกัน โดยทั้ง 5 กิจกรรมยังไม่มีการซื้อ")
st.caption("ราคาต้องใช้หน่วยเดียวกับข้อมูลที่ใช้ฝึกโมเดล")

features = {}
col1, col2 = st.columns(2)

with col1:
    features["cart_count_first5"] = st.number_input(
        "จำนวนครั้งที่เพิ่มสินค้าลงตะกร้า", min_value=0, max_value=5, value=1
    )
    features["unique_products_first5"] = st.number_input(
        "จำนวนสินค้าที่ไม่ซ้ำกัน (รายการ)", min_value=1, max_value=5, value=2,
        help="นับสินค้าที่ดูหรือเพิ่มลงตะกร้าใน 5 กิจกรรมแรก ถ้าเป็นสินค้าเดิมให้นับเป็น 1 รายการ"
    )
    features["unique_categories_first5"] = st.number_input(
        "จำนวนหมวดหมู่สินค้า (หมวด)", min_value=1, max_value=5, value=1,
        help="นับหมวดหมู่ที่ไม่ซ้ำกันจากทั้ง 5 กิจกรรม"
    )
    minutes = st.number_input(
        "เวลาจากกิจกรรมแรกถึงกิจกรรมที่ 5 (นาที)", min_value=0.0, value=2.0
    )
    features["time_span_first5"] = minutes * 60

with col2:
    features["mean_price_first5"] = st.number_input(
        "ราคาเฉลี่ยใน 5 กิจกรรมแรก", min_value=0.0, value=180.0
    )
    features["max_price_first5"] = st.number_input(
        "ราคาสินค้าสูงสุด", min_value=0.0, value=200.0
    )
    features["first_event_type"] = st.selectbox(
        "กิจกรรมแรก", ["view", "cart"], format_func=activities.get,
        help="สิ่งที่ลูกค้าทำเป็นลำดับแรกในการเข้าเว็บครั้งนี้"
    )
    features["fifth_event_type"] = st.selectbox(
        "กิจกรรมที่ 5", ["view", "cart"], index=1, format_func=activities.get,
        help="สิ่งที่ลูกค้าทำเป็นลำดับที่ 5 การดูสินค้าเดิมซ้ำก็นับเป็นอีก 1 กิจกรรม"
    )

# ตรวจข้อมูล แล้วทำนายเมื่อกดปุ่ม
if st.button("ดูผลทำนาย", type="primary"):
    end_types = [
        features["first_event_type"],
        features["fifth_event_type"]
    ]
    end_carts = end_types.count("cart")
    cart_count = features["cart_count_first5"]
    mean_price = features["mean_price_first5"]
    max_price = features["max_price_first5"]

    if cart_count < end_carts or cart_count > end_carts + 3:
        st.error("จำนวนครั้งที่เพิ่มลงตะกร้าไม่ตรงกับกิจกรรมแรกและกิจกรรมที่ 5")
    elif max_price < mean_price or max_price > 5 * mean_price:
        st.error("ราคาสูงสุดต้องเท่ากับหรือมากกว่าราคาเฉลี่ย และต้องไม่เกิน 5 เท่าของราคาเฉลี่ย")
    else:
        prediction, score = predict(features)

        # สรุปคะแนนและผลทำนายในกรอบเดียว
        with st.container(border=True):
            st.subheader("ผลการทำนาย")
            score_col, result_col = st.columns([1, 2])

            with score_col:
                st.metric("คะแนนจากโมเดล", f"{score:.1%}")
                st.progress(float(score))

            with result_col:
                if prediction == 1:
                    st.success("คาดว่าจะซื้อสินค้า")
                else:
                    st.info("คาดว่าจะไม่ซื้อสินค้า")
                st.caption("ผลทำนายการซื้อหลังจาก 5 กิจกรรมแรก")

            st.caption(f"โมเดลทำนายว่าจะซื้อ เมื่อคะแนนมากกว่า {threshold:.0%}")

        # แสดงข้อมูลที่เพิ่มและลดคะแนน สูงสุดด้านละ 3 รายการ
        factors = explain(features)
        positive = factors[factors["Effect"] > 0.000001]
        positive = positive.sort_values("Effect", ascending=False).head(3)
        negative = factors[factors["Effect"] < -0.000001]
        negative = negative.sort_values("Effect").head(3)

        # แยกข้อมูลที่เพิ่มและลดคะแนนเป็นสองกลุ่ม
        st.subheader("ข้อมูลที่มีผลต่อคะแนน")
        st.caption("เรียงจากรายการที่มีผลมากที่สุด แสดงสูงสุดด้านละ 3 รายการ")
        positive_col, negative_col = st.columns(2)

        with positive_col:
            with st.container(border=True):
                st.markdown("**↑ ช่วยเพิ่มคะแนน**")
                if positive.empty:
                    st.caption("ไม่มีรายการในกลุ่มนี้")
                for text in positive["Factor"]:
                    st.markdown("- " + text)

        with negative_col:
            with st.container(border=True):
                st.markdown("**↓ ทำให้คะแนนลดลง**")
                if negative.empty:
                    st.caption("ไม่มีรายการในกลุ่มนี้")
                for text in negative["Factor"]:
                    st.markdown("- " + text)

        # เปิดอ่านรายละเอียดการเปรียบเทียบได้เมื่อสนใจ
        with st.expander("ข้อมูลเหล่านี้เปรียบเทียบกับอะไร?"):
            st.write("จำนวนสินค้า จำนวนครั้งที่เพิ่มลงตะกร้า ราคา และเวลา: เทียบกับค่าเฉลี่ยในข้อมูลที่ใช้ฝึกโมเดล")
            st.write("กิจกรรมแรกและกิจกรรมที่ 5: เทียบกับการดูสินค้า")
