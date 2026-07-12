import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.interpolate import CubicSpline
from datetime import datetime
import os
import re

st.set_page_config(page_title='Compressor Curve Regression', layout='wide')
st.title('Compressor Curve Regression Tool')

method = st.sidebar.selectbox('Regression Method',
    ['Auto Best Fit','Linear','Quadratic','Cubic','4th Order','5th Order','Spline'])
points = st.sidebar.slider('Generated Points',15,50,15)
file = st.file_uploader('Upload Workbook', type=['xlsx'])

# ---------------------------------------------------------------------------
# Physical constants and unit conversions
# ---------------------------------------------------------------------------
R_UNIVERSAL = 8314.462618   # J/(kmol.K)
G = 9.80665                 # m/s^2
P_ATM_KG_CM2 = 1.033227     # Standard Atmospheric Pressure in kg/cm2

def normalize_unit(u):
    """Fold formatting variants ('m³/hr', 'M3/HR', 'ft^3 / min', 'Ft3.Min') onto one key."""
    s = str(u).strip().lower()
    s = s.replace('³', '3').replace('²', '2')
    s = s.replace('^3', '3').replace('^2', '2')
    s = s.replace(' ', '').replace('.', '').replace('-', '').replace('_', '')
    s = s.replace('cu', '')  # 'cuft' -> 'ft'
    return s

# All target units: Flow -> m3/hr, Head -> m, Power -> kW, Efficiency -> %
FLOW_TO_M3HR = {
    'cfm': 1.699010796, 'acfm': 1.699010796, 'icfm': 1.699010796,
    'ft3/min': 1.699010796, 'ft3min': 1.699010796, 'ft/min': 1.699010796, 'cf/min': 1.699010796,
    'cfh': 0.028316847, 'ft3/hr': 0.028316847, 'ft3/h': 0.028316847, 'ft3hr': 0.028316847,
    'ft/hr': 0.028316847, 'cf/hr': 0.028316847,
    'cfs': 101.9406, 'ft3/s': 101.9406, 'ft3s': 101.9406, 'ft/s': 101.9406, 'cf/s': 101.9406,
    'm3/hr': 1.0, 'm3/h': 1.0, 'm3hr': 1.0, 'm3h': 1.0,
    'm3/min': 60.0, 'm3min': 60.0,
    'm3/s': 3600.0, 'm3s': 3600.0,
    'l/min': 0.06, 'lpm': 0.06,
    'l/s': 3.6, 'lps': 3.6, 'l/hr': 0.001, 'lph': 0.001,
    'gpm': 0.227124707, 'usgpm': 0.227124707, 'galmin': 0.227124707,
    'igpm': 0.272765, 'ukgpm': 0.272765, 'impgpm': 0.272765,
    'gph': 0.003785412, 'usgph': 0.003785412,
    'bbl/day': 0.006624459, 'bpd': 0.006624459, 'bbl/d': 0.006624459,
    'mmscfd': 1179.874,
}

HEAD_TO_M = {
    'ft': 0.3048, 'feet': 0.3048, 'foot': 0.3048,
    'ftlbf/lbm': 0.3048, 'lbfft/lbm': 0.3048, 'ftlb/lb': 0.3048,
    'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
    'm': 1.0, 'meter': 1.0, 'metre': 1.0, 'meters': 1.0, 'metres': 1.0,
    'mm': 0.001,
    'kj/kg': 101.9716, 'j/kg': 0.1019716,
    'btu/lb': 237.2075,
}

POWER_TO_KW = {
    'hp': 0.745699872, 'bhp': 0.745699872, 'mechhp': 0.745699872, 'hp(i)': 0.745699872,
    'ps': 0.735499, 'cv': 0.735499, 'metrichp': 0.735499, 'hp(m)': 0.735499,
    'kw': 1.0, 'w': 0.001, 'mw': 1000.0,
    'btu/hr': 0.000293071, 'btu/h': 0.000293071, 'btuh': 0.000293071,
    'btu/s': 1.055056, 'btus': 1.055056,
    'ftlb/s': 0.001355818, 'ftlbf/s': 0.001355818,
    'kcal/hr': 0.001163, 'kcal/h': 0.001163,
}

EFF_TO_PCT = {
    '%': 1.0, 'pct': 1.0, 'percent': 1.0, 'percentage': 1.0,
    'fraction': 100.0, 'decimal': 100.0, 'ratio': 100.0, 'frac': 100.0,
}

DIAMETER_TO_M = {
    'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
    'mm': 0.001, 'milimeter': 0.001, 'milimeters': 0.001,
    'cm': 0.01, 'centimeter': 0.01,
    'm': 1.0, 'meter': 1.0, 'meters': 1.0
}

def convert_temperature_to_c(val, unit_str):
    """Handles temperature offset scales directly instead of single scalar multipliers."""
    u = str(unit_str).strip().lower()
    u = u.replace('°', '').replace('degree', '').replace('deg', '')
    u = u.replace(' ', '').replace('.', '').replace('-', '').replace('_', '')
    
    if u in ['f', 'fahrenheit']:
        return (val - 32) * 5.0 / 9.0, True
    if u in ['k', 'kelvin']:
        return val - 273.15, True
    if u in ['r', 'rankine']:
        return (val - 491.67) * 5.0 / 9.0, True
    if u in ['c', 'celsius', 'centigrade']:
        return val, True
    return val, False

def convert_pressure_to_kg_cm2a(val, unit_str):
    """Handles scalar conversion for pressure and scales up gauge options to kg/cm2 absolute."""
    u = normalize_unit(unit_str)
    
    multipliers = {
        'psi': 0.070306958, 'psia': 0.070306958, 'psig': 0.070306958,
        'bar': 1.01971621, 'bara': 1.01971621, 'barg': 1.01971621,
        'kpa': 0.010197162, 'kpaa': 0.010197162, 'kpag': 0.010197162,
        'mpa': 10.1971621, 'mpaa': 10.1971621, 'mpag': 10.1971621,
        'kg/cm2': 1.0, 'kg/cm2a': 1.0, 'kg/cm2g': 1.0, 
        'kgf/cm2': 1.0, 'kgf/cm2a': 1.0, 'kgf/cm2g': 1.0,
        'atm': 1.033227, 'atmg': 1.033227, 'pa': 0.00001019716, 'pag': 0.00001019716
    }
    
    if u not in multipliers:
        return val, False
        
    kg_cm2_val = val * multipliers[u]
    
    if u.endswith('g') or u in ['psi', 'bar', 'kpa', 'mpa', 'kg/cm2', 'kgf/cm2']:
        if not u.endswith('a'):
            return kg_cm2_val + P_ATM_KG_CM2, True
            
    return kg_cm2_val, True

def convert_unit(value, unit_str, table, label):
    key = normalize_unit(unit_str)
    if key in table:
        return value * table[key], True
    return value, False

def kg_cm2a_to_pa(kg_cm2a):
    return kg_cm2a * 98066.5

def c_to_k(deg_c):
    return deg_c + 273.15

def gas_density_kg_m3(pressure_kg_cm2a, temperature_c, mw, z):
    """Inlet gas density using metric units: rho = P*MW / (Z*R*T)."""
    p_pa = kg_cm2a_to_pa(pressure_kg_cm2a)
    t_k = c_to_k(temperature_c)
    return (p_pa * mw) / (z * R_UNIVERSAL * t_k)

def clean_parameter_name(name):
    n = str(name).lower()
    if 'head' in n: return 'Head'
    if 'eff' in n: return 'Efficiency'
    if 'power' in n or 'bhp' in n or 'kw' in n: return 'Power'
    return str(name)

def detect_triplet_blocks(raw_df):
    blocks = []
    rows, cols = raw_df.shape
    for r in range(rows):
        for c in range(cols - 2):
            v1 = str(raw_df.iloc[r, c]).strip().lower()
            v2 = str(raw_df.iloc[r, c + 1]).strip().lower()
            v3 = str(raw_df.iloc[r, c + 2]).strip()
            if v1 == 'speed' and 'flow' in v2:
                p = clean_parameter_name(v3)
                if p.lower() != 'nan':
                    speed_unit = flow_unit = value_unit = ''
                    if r + 1 < rows:
                        speed_unit = str(raw_df.iloc[r + 1, c]).strip()
                        flow_unit = str(raw_df.iloc[r + 1, c + 1]).strip()
                        value_unit = str(raw_df.iloc[r + 1, c + 2]).strip()
                    blocks.append({
                        'parameter': p, 'header_row': r, 'start_col': c,
                        'speed_unit': speed_unit, 'flow_unit': flow_unit,
                        'value_unit': value_unit
                    })
    uniq = []
    seen = set()
    for b in blocks:
        k = b['start_col']
        if k not in seen:
            seen.add(k)
            uniq.append(b)
    return uniq

def extract_block_data(raw_df, block):
    r = block['header_row']
    c = block['start_col']
    data = []
    for row in range(r + 1, len(raw_df)):
        try:
            sp = float(raw_df.iloc[row, c])
            fl = float(raw_df.iloc[row, c + 1])
            val = float(raw_df.iloc[row, c + 2])
            data.append([sp, fl, val])
        except:
            pass
    df = pd.DataFrame(data, columns=['Speed', 'Flow', 'Value'])
    if df.empty:
        return df, False, False

    flow_converted = value_converted = False
    df['Flow'], flow_converted = convert_unit(df['Flow'], block['flow_unit'], FLOW_TO_M3HR, 'flow')

    param = block['parameter']
    if param == 'Head':
        df['Value'], value_converted = convert_unit(df['Value'], block['value_unit'], HEAD_TO_M, 'head')
    elif param == 'Power':
        df['Value'], value_converted = convert_unit(df['Value'], block['value_unit'], POWER_TO_KW, 'power')
    elif param == 'Efficiency':
        df['Value'], value_converted = convert_unit(df['Value'], block['value_unit'], EFF_TO_PCT, 'efficiency')

    return df, flow_converted, value_converted

def detect_property_block(raw_df):
    rows, cols = raw_df.shape
    for r in range(rows):
        for c in range(cols - 2):
            v1 = str(raw_df.iloc[r, c]).strip().lower()
            v2 = str(raw_df.iloc[r, c + 1]).strip().lower()
            v3 = str(raw_df.iloc[r, c + 2]).strip().lower()
            if v1 == 'parameter' and v2 == 'value' and v3 == 'units':
                return {'header_row': r, 'start_col': c}
    return None

def extract_property_block(raw_df, block):
    if block is None:
        return pd.DataFrame(columns=['Parameter', 'Value', 'Units'])
    r = block['header_row']
    c = block['start_col']
    rows_out = []
    for row in range(r + 1, len(raw_df)):
        param = raw_df.iloc[row, c]
        value = raw_df.iloc[row, c + 1]
        units = raw_df.iloc[row, c + 2]
        if pd.isna(param) or str(param).strip() == '':
            break
        
        p_name = str(param).strip()
        v_val = value.item() if hasattr(value, 'item') else value
        u_str = '' if pd.isna(units) else str(units).strip()
        
        if 'diameter' in p_name.lower():
            try:
                if isinstance(v_val, str):
                    v_val = re.split(r'[,/]', v_val)[0].strip()
                
                converted_val, success = convert_unit(float(v_val), u_str, DIAMETER_TO_M, 'diameter')
                if success:
                    v_val = converted_val
                    u_str = 'm'
            except (ValueError, TypeError):
                pass
        elif 'pressure' in p_name.lower():
            try:
                if isinstance(v_val, str):
                    v_val = re.split(r'[,/]', v_val)[0].strip()
                
                converted_val, success = convert_pressure_to_kg_cm2a(float(v_val), u_str)
                if success:
                    v_val = converted_val
                    u_str = 'kg/cm2a'
            except (ValueError, TypeError):
                pass
        elif 'temperature' in p_name.lower():
            try:
                if isinstance(v_val, str):
                    v_val = re.split(r'[,/]', v_val)[0].strip()
                
                converted_val, success = convert_temperature_to_c(float(v_val), u_str)
                if success:
                    v_val = converted_val
                    u_str = 'deg C'
            except (ValueError, TypeError):
                pass

        rows_out.append({
            'Parameter': p_name,
            'Value': v_val,
            'Units': u_str
        })
    return pd.DataFrame(rows_out, columns=['Parameter', 'Value', 'Units'])

def build_model(x, y, meth):
    if meth == 'Spline':
        idx = np.argsort(x)
        x = x[idx]; y = y[idx]
        s = CubicSpline(x, y)
        r2 = r2_score(y, s(x))
        return {'type': 'spline', 'model': s, 'xmin': x.min(), 'xmax': x.max(), 'r2': r2}

    deg = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3, '4th Order': 4, '5th Order': 5}[meth]
    poly = PolynomialFeatures(deg)
    X = poly.fit_transform(x.reshape(-1, 1))
    lr = LinearRegression().fit(X, y)
    r2 = r2_score(y, lr.predict(X))
    return {'type': 'poly', 'poly': poly, 'model': lr, 'xmin': x.min(), 'xmax': x.max(), 'r2': r2}

def predict_model(obj, flow):
    if obj['type'] == 'spline':
        return obj['model'](flow)
    return obj['model'].predict(obj['poly'].transform(flow.reshape(-1, 1)))

def auto_best(x, y):
    best = None; best_name = None; best_r2 = -1e9
    for m in ['Linear', 'Quadratic', 'Cubic', '4th Order', '5th Order', 'Spline']:
        try:
            mdl = build_model(x, y, m)
            if mdl['r2'] > best_r2:
                best_r2 = mdl['r2']; best = mdl; best_name = m
        except:
            pass
    return best_name, best

def gas_properties_from_df(prop_df):
    lookup = {}
    for _, row in prop_df.iterrows():
        name = str(row['Parameter']).lower()
        if 'pressure' in name:
            lookup['pressure'] = row['Value']
        elif 'temperature' in name:
            lookup['temperature'] = row['Value']
        elif 'molecular weight' in name or 'mw' in name:
            lookup['mw'] = row['Value']
        elif 'compressibility' in name or 'z' in name:
            lookup['z'] = row['Value']
    try:
        return {
            'pressure_kg_cm2a': float(lookup['pressure']),
            'temperature_c': float(lookup['temperature']),
            'mw': float(lookup['mw']),
            'z': float(lookup['z']),
        }
    except (KeyError, TypeError, ValueError):
        return None

def compute_missing_parameter(available, mass_flow_kg_s):
    have = set(available.keys())
    needed = {'Head', 'Efficiency', 'Power'} - have
    if len(needed) != 1:
        return None, None
    missing = needed.pop()

    if missing == 'Power':
        eff = available['Efficiency'] / 100.0
        power_kw = mass_flow_kg_s * G * available['Head'] / eff / 1000.0
        return 'Power', power_kw

    if missing == 'Head':
        eff = available['Efficiency'] / 100.0
        head_m = available['Power'] * 1000.0 * eff / (mass_flow_kg_s * G)
        return 'Head', head_m

    if missing == 'Efficiency':
        eff_pct = (mass_flow_kg_s * G * available['Head']) / (available['Power'] * 1000.0) * 100.0
        return 'Efficiency', eff_pct

    return None, None


if file:
    xls = pd.ExcelFile(file)
    output = BytesIO()
    r2_rows = []
    overview = []
    property_rows = []
    fatal_error = None

    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame([{'status': 'processing started'}]).to_excel(
                writer, sheet_name='_status', index=False)

            for stage in xls.sheet_names:
                st.header(stage)
                try:
                    raw = pd.read_excel(file, sheet_name=stage, header=None)
                    blocks = detect_triplet_blocks(raw)

                    prop_block = detect_property_block(raw)
                    prop_df = extract_property_block(raw, prop_block)
                    gas_props = None
                    if not prop_df.empty:
                        st.subheader('Operating Conditions')
                        st.dataframe(prop_df, use_container_width=True)
                        for _, row in prop_df.iterrows():
                            property_rows.append([stage, row['Parameter'], row['Value'], row['Units']])
                        gas_props = gas_properties_from_df(prop_df)
                        if gas_props is None:
                            st.info('Could not read Pressure/Temperature/MW/Compressibility as numbers — '
                                    'skipping missing-parameter calculation for this stage.')
                    else:
                        st.warning(f'No operating-conditions block found in {stage}')

                    if not blocks:
                        st.warning(f'No blocks found in {stage}')
                        overview.append({'Stage': stage, 'Parameters': '', 'Blocks Found': 0,
                                          'Calculated Parameter': '', 'Status': 'no blocks found'})
                        continue

                    clean_blocks = []
                    for b in blocks:
                        looks_numeric = False
                        for u in (b['speed_unit'], b['flow_unit'], b['value_unit']):
                            try:
                                float(u)
                                looks_numeric = True
                            except (ValueError, TypeError):
                                pass
                        if looks_numeric:
                            st.warning(f"Skipping a detected block for '{b['parameter']}' at column "
                                       f"{b['start_col']} — its units row looks like data, not units "
                                       f"('{b['speed_unit']}', '{b['flow_unit']}', '{b['value_unit']}'). "
                                       f"This usually means a duplicated/mislabeled header row in the source sheet.")
                        else:
                            clean_blocks.append(b)
                    blocks = clean_blocks

                    if not blocks:
                        st.warning(f'No valid blocks remained after validation in {stage}')
                        overview.append({'Stage': stage, 'Parameters': '', 'Blocks Found': 0,
                                          'Calculated Parameter': '', 'Status': 'blocks failed validation'})
                        continue

                    block_summary = pd.DataFrame([
                        {'parameter': b['parameter'], 'speed_unit': b['speed_unit'],
                         'flow_unit': b['flow_unit'], 'value_unit': b['value_unit']}
                        for b in blocks
                    ])
                    st.dataframe(block_summary)

                    stage_models = {}
                    stage_parameters = []

                    tabs = st.tabs([b['parameter'] for b in blocks])

                    for tab, block in zip(tabs, blocks):
                        with tab:
                            param = block['parameter']
                            df, flow_conv, val_conv = extract_block_data(raw, block)
                            stage_parameters.append(param)

                            unit_note = []
                            if not flow_conv:
                                unit_note.append(f"flow unit '{block['flow_unit']}' not recognized, left as-is")
                            if not val_conv:
                                unit_note.append(f"{param} unit '{block['value_unit']}' not recognized, left as-is")
                            if unit_note:
                                st.caption('⚠ ' + '; '.join(unit_note))
                            else:
                                target_unit = {'Head': 'm', 'Power': 'kW', 'Efficiency': '%'}.get(param, '')
                                st.caption(f"Converted to Flow: m³/hr, {param}: {target_unit}")

                            fig = go.Figure()

                            if param not in stage_models:
                                stage_models[param] = {}

                            for speed in sorted(df['Speed'].unique()):
                                sdf = df[df['Speed'] == speed]
                                if len(sdf) < 4:
                                    continue

                                x = sdf['Flow'].values.astype(float)
                                y = sdf['Value'].values.astype(float)

                                if method == 'Auto Best Fit':
                                    used, mdl = auto_best(x, y)
                                else:
                                    mdl = build_model(x, y, method)
                                    used = method

                                if mdl is None:
                                    continue

                                stage_models[param][speed] = mdl

                                r2_rows.append([stage, speed, param, used, round(mdl['r2'], 6)])

                                flow_fit = np.linspace(x.min(), x.max(), points)
                                y_fit = predict_model(mdl, flow_fit)

                                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{speed} Original'))
                                fig.add_trace(go.Scatter(x=flow_fit, y=y_fit, mode='lines', name=f'{speed} Fit'))

                            st.plotly_chart(fig, use_container_width=True)

                    speeds = set()
                    for p in stage_models:
                        speeds.update(stage_models[p].keys())

                    export_rows = []
                    computed_param_name = None
                    stage_status = 'ok'

                    for speed in sorted(speeds):
                        available = []
                        available_params = []
                        for p in stage_models:
                            if speed in stage_models[p]:
                                available.append(stage_models[p][speed])
                                available_params.append(p)

                        if len(available) < 1:
                            continue

                        common_min = max(m['xmin'] for m in available)
                        common_max = min(m['xmax'] for m in available)

                        # ERROR INTERACTION CHECK: Flag if flow ranges do not overlap
                        if common_max <= common_min:
                            err_msg = f"Flow values do not overlap for Stage: **{stage}** at Speed: **{speed}** across parameters ({', '.join(available_params)})."
                            st.error(err_msg)
                            stage_status = f"error: flow overlap failed at speed {speed}"
                            continue

                        common_flow = np.linspace(common_min, common_max, points)

                        temp = {'Speed': [speed] * points, 'Flow (m3/hr)': common_flow}

                        predicted = {}
                        for p in stage_models:
                            if speed in stage_models[p]:
                                vals = predict_model(stage_models[p][speed], common_flow)
                                predicted[p] = vals
                                unit_label = {'Head': 'm', 'Power': 'kW', 'Efficiency': '%'}.get(p, '')
                                temp[f'{p} ({unit_label})'] = vals

                        if gas_props is not None:
                            try:
                                rho = gas_density_kg_m3(gas_props['pressure_kg_cm2a'], gas_props['temperature_c'],
                                                         gas_props['mw'], gas_props['z'])
                                mass_flow_kg_s = common_flow * rho / 3600.0
                                name, values = compute_missing_parameter(predicted, mass_flow_kg_s)
                                if name is not None:
                                    computed_param_name = name
                                    unit_label = {'Head': 'm', 'Power': 'kW', 'Efficiency': '%'}.get(name, '')
                                    temp[f'{name} ({unit_label}, calculated)'] = values
                            except (ZeroDivisionError, ValueError, KeyError) as e:
                                st.warning(f"Could not compute missing parameter for {stage} @ speed {speed}: {e}")

                        export_rows.append(pd.DataFrame(temp))

                    if computed_param_name and stage_status == 'ok':
                        st.success(f"Calculated missing parameter **{computed_param_name}** for {stage} "
                                   f"using gas density from Operating Conditions.")

                    if export_rows:
                        final_df = pd.concat(export_rows, ignore_index=True)
                        final_df.to_excel(writer, sheet_name=stage[:31], index=False)

                    overview.append({
                        'Stage': stage,
                        'Parameters': ','.join(stage_parameters),
                        'Blocks Found': len(blocks),
                        'Calculated Parameter': computed_param_name or '',
                        'Status': stage_status
                    })

                except Exception as e:
                    st.error(f"Error processing '{stage}': {e}")
                    overview.append({'Stage': stage, 'Parameters': '', 'Blocks Found': 0,
                                      'Calculated Parameter': '', 'Status': f'error: {e}'})

            pd.DataFrame(r2_rows, columns=['Stage', 'Speed', 'Parameter', 'Method', 'R2']).to_excel(
                writer, sheet_name='Summary_R2', index=False)
            pd.DataFrame(overview).to_excel(writer, sheet_name='Workbook_Overview', index=False)

            if property_rows:
                pd.DataFrame(
                    property_rows, columns=['Stage', 'Parameter', 'Value', 'Units']
                ).to_excel(writer, sheet_name='Operating_Conditions', index=False)

    except Exception as e:
        fatal_error = e

    if fatal_error is not None:
        st.error(f"Could not build the output workbook: {fatal_error}")
        st.info("Nothing to download — see the error above.")
    else:
        output.seek(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = os.path.splitext(file.name)[0]
        output_filename = f"{input_filename}_Regression_Output_{timestamp}.xlsx"

        st.download_button(
            label="Download Regression Workbook",
            data=output.getvalue(),
            file_name=output_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
