import streamlit as st
import numpy as np
from gwpy.timeseries import TimeSeries # type: ignore
from scipy.signal import welch # type: ignore

st.title("GravNet: GW Data Fetcher & Whitening")

st.markdown("""
Use the selectors in the sidebar to pick a start and end GPS time, then:
1. **Fetch** the strain data  
2. **Whiten** the signal
3. **Extract Noise** from the signal
4. **Detect** Gravitational Wave Signal
5. **Estimate** source Parameters
""")

st.sidebar.header("GPS Time Selection")
t0 = st.sidebar.number_input(
    "Start GPS time",
    min_value=0.0,
    value=1126259461.472,
    step=0.001,
    format="%.3f",
    help="GPS seconds since Jan 1, 1970 UTC"
)
t1 = t0 + 1.0

if st.sidebar.button("Fetch Data"):
    with st.spinner(f"Fetching data from {t0} to {t1}..."):
        try:
            strain = TimeSeries.fetch_open_data('L1', t0, t1, cache=True)
            st.success("Data loaded!")
            st.session_state['strain'] = strain

            st.subheader("Raw Strain")
            st.line_chart(strain.value)

            st.markdown("**Data summary:**")
            st.write(f"- Duration: {strain.duration.value:.2f} s")
            st.write(f"- Sampling rate: {4096:.0f} Hz")
            st.write(f"- Min/Max strain: {strain.value.min():.2e} / {strain.value.max():.2e}")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")

if "strain" in st.session_state:
    st.sidebar.header("Signal Processing")
    if st.sidebar.button("Whiten Signal"):
        strain: TimeSeries = st.session_state["strain"]
        with st.spinner("Running whitening..."):
            frequencies, psd_strain = welch(strain.value, fs=1/4096, nperseg=4096)
            freq_template = np.fft.rfftfreq(len(strain), 4096)
            psd_interp = np.interp(freq_template, frequencies, psd_strain)
            whitened_strain = np.fft.irfft(np.fft.rfft(strain.value)/psd_interp**0.5).real
            st.session_state["whitened_strain"] = whitened_strain
            
            st.subheader("Whitened Strain")
            st.success("Data Whitened!")
            st.line_chart(whitened_strain)

if "whitened_strain" in st.session_state:
    import torch
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    labels = ["Noise", "BBH Merger", "BNS Merger"]

    st.sidebar.header("Noise Extraction")
    if st.sidebar.button("UNet Noise Extraction"):
        from gravnet.unet import UNet
        
        model = UNet([1, 32, 64, 128, 256, 512], kernel_size=3).to(device)
        state = torch.load("model_weights/unet_noise_extr.pth", map_location=device)
        model.load_state_dict(state["weights_only"] if "weights_only" in state else state)

        whitened_strain = st.session_state["whitened_strain"]
        input_signal = torch.tensor(whitened_strain, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_signal).squeeze().detach().cpu().numpy()

        st.subheader("Extracted Noise")
        st.line_chart(output)

    st.sidebar.header("Wave Detection")
    if st.sidebar.button("CNN Wave Detection"):
        from gravnet.cnn_cls_reg import CNNClsReg

        model = CNNClsReg(task="classification").to(device)
        state = torch.load("model_weights/cnn_classifier.pth", map_location=device)
        model.load_state_dict(state["weights_only"] if "weights_only" in state else state)
        
        whitened_strain = st.session_state["whitened_strain"]
        input_signal = torch.tensor(whitened_strain, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_signal).squeeze().detach().cpu().numpy()

        st.subheader("CNN Output")
        st.write(labels[np.argmax(output)])
        st.write(output)
        st.line_chart(input_signal.squeeze().cpu().numpy())

    if st.sidebar.button("ResNet Wave Detection"):
        from gravnet.resnet import ResNetClassifier

        model = ResNetClassifier(in_channels=1, num_classes=3).to(device)
        state = torch.load("model_weights/resnet_cls.pth", map_location=device)
        model.load_state_dict(state["weights_only"] if "weights_only" in state else state)

        whitened_strain = st.session_state["whitened_strain"]
        input_signal = torch.tensor(whitened_strain, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_signal.unsqueeze(1)).squeeze().detach().cpu().numpy()

        st.subheader("ResNet Output")
        st.write(labels[np.argmax(output)])
        st.write(output)
        st.line_chart(input_signal.squeeze().cpu().numpy())

    if st.sidebar.button("DenseNet Wave Detection"):
        from gravnet.densenet import DenseNet

        model = DenseNet(in_channels=1, num_params=3).to(device)
        state = torch.load("model_weights/densenet_cls.pth", map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

        whitened_strain = st.session_state["whitened_strain"]
        input_signal = torch.tensor(whitened_strain, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_signal.unsqueeze(1)).squeeze().detach().cpu().numpy()

        st.subheader("DenseNet Output")
        st.write(labels[np.argmax(output)])
        st.write(output)
        st.line_chart(input_signal.squeeze().cpu().numpy())

    if st.sidebar.button("UNet Wave Detection"):
        from gravnet.unet import UNetFineTuned

        model = UNetFineTuned(backbone_path="model_weights/unet_noise_extr.pth", device=device).to(device)
        state = torch.load("model_weights/unet_classifier.pth", map_location=device)
        model.load_state_dict(state["weights_only"] if "weights_only" in state else state)

        whitened_strain = st.session_state["whitened_strain"]
        input_signal = torch.tensor(whitened_strain, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_signal).squeeze().detach().cpu().numpy()

        st.subheader("UNet Output")
        st.write(labels[np.argmax(output)])
        st.write(output)
        st.line_chart(input_signal.squeeze().cpu().numpy())

    st.sidebar.header("Parameter Estimation")
    if st.sidebar.button("CNN Parameter Estimation"):
        from gravnet.cnn_cls_reg import CNNClsReg

        model = CNNClsReg(task="regression").to(device)
        state = torch.load("model_weights/cnn_regressor.pth", map_location=device)
        model.load_state_dict(state["weights_only"] if "weights_only" in state else state)

        whitened_strain = st.session_state["whitened_strain"]
        input_signal = torch.tensor(whitened_strain, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_signal).squeeze().detach().cpu().numpy()

        st.subheader("CNN Output")
        st.markdown(f"""
            $m_1$ = {output[0]:.4f}
            $m_2$ = {output[1]:.4f}
            SNR = {output[2]:.4f}
        """)

    if st.sidebar.button("ResNet Parameter Estimation"):
        from gravnet.resnet import ResNet

        model = ResNet(in_channels=1, num_params=3).to(device)
        state = torch.load("model_weights/resnet_reg.pth", map_location=device)
        model.load_state_dict(state["weights_only"] if "weights_only" in state else state)

        whitened_strain = st.session_state["whitened_strain"]
        input_signal = torch.tensor(whitened_strain, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_signal.unsqueeze(1)).squeeze().detach().cpu().numpy()

        st.subheader("ResNet Output")
        st.markdown(f"""
            $m_1$ = {output[0]:.4f}
            $m_2$ = {output[1]:.4f}
            SNR = {output[2]:.4f}
        """)

    if st.sidebar.button("DenseNet Parameter Estimation"):
        from gravnet.densenet import DenseNet

        model = DenseNet(in_channels=1, num_params=3).to(device)
        state = torch.load("model_weights/densenet_cls.pth", map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

        whitened_strain = st.session_state["whitened_strain"]
        input_signal = torch.tensor(whitened_strain, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_signal.unsqueeze(1)).squeeze().detach().cpu().numpy()

        st.subheader("DenseNet Output")
        st.markdown(f"""
            $m_1$ = {output[0]:.4f}
            $m_2$ = {output[1]:.4f}
            SNR = {output[2]:.4f}
        """)

    if st.sidebar.button("UNet Parameter Estimation"):
        from gravnet.unet import UNetFineTuned

        model = UNetFineTuned(backbone_path="model_weights/unet_noise_extr.pth", device=device)
        state = torch.load("model_weights/unet_regressor.pth", map_location=device)
        model.load_state_dict(state["weights_only"] if "weights_only" in state else state)

        whitened_strain = st.session_state["whitened_strain"]
        input_signal = torch.tensor(whitened_strain, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_signal).squeeze().detach().cpu().numpy()

        st.subheader("UNet Output")
        st.markdown(f"""
            $m_1$ = {output[0]:.4f}
            $m_2$ = {output[1]:.4f}
            SNR = {output[2]:.4f}
        """)
