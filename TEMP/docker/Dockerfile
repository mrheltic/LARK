# ============================================================
#  KrakenSDR – Multi-stage Docker Image
#
#  Stage 1 "builder"  – compile all C/C++ components
#  Stage 2 "runtime"  – lean image without build tools
#
#  Build-time ARGs (override via --build-arg):
#    UBUNTU_VERSION   Ubuntu release tag          (default: 22.04)
#    KFR_VERSION      kfrlib/kfr git tag          (default: 5.0.2)
#    BUILD_FLAGS      C/C++ optimisation flags    (default: -O2 -pipe)
#    GIT_COMMIT       Short commit hash for label (default: dev)
# ============================================================

ARG UBUNTU_VERSION=22.04

# ════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — builder
#  All build-tools, compilers and header packages stay here.
#  Nothing in this stage ends up in the final image.
# ════════════════════════════════════════════════════════════════════════════
FROM ubuntu:${UBUNTU_VERSION} AS builder

ARG KFR_VERSION=5.0.2
ARG BUILD_FLAGS="-O2 -pipe"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -o Acquire::Retries=5 update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config clang ca-certificates \
    libusb-1.0-0-dev \
    libzmq3-dev \
    gnuradio-dev \
    python3 python3-dev python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/krakensdr
COPY gr-krakensdr/    ./gr-krakensdr/
COPY heimdall_daq_fw/ ./heimdall_daq_fw/

# ── 1a. Build krakenrf/librtlsdr from the bundled submodule ─────────────────
#        The krakenrf fork patches multi-dongle clock synchronisation.
#        The Heimdall Makefile links with "-L. -lrtlsdr"; after installing
#        to /usr/local the linker finds it via ldconfig.
RUN cmake -S /opt/krakensdr/heimdall_daq_fw/Firmware/librtlsdr \
          -B /tmp/librtlsdr-build \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_FLAGS="${BUILD_FLAGS}" \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DINSTALL_UDEV_RULES=OFF && \
    make -C /tmp/librtlsdr-build -j"$(nproc)" install && \
    # The krakenrf cmake names the static target "rtlsdr_static" → librtlsdr_static.a
    # on Linux (OUTPUT_NAME rtlsdr is only set on WIN32).  The Heimdall Makefile
    # links with "-L. -lrtlsdr", which requires librtlsdr.a in _daq_core/.
    # Copy the static archive under the expected name so the linker finds it.
    STATIC=$(find /tmp/librtlsdr-build -name 'librtlsdr*.a' | head -1) && \
    [ -n "$STATIC" ] && \
    cp "$STATIC" /opt/krakensdr/heimdall_daq_fw/Firmware/_daq_core/librtlsdr.a && \
    echo "[✓] librtlsdr.a → _daq_core/ (from $STATIC)" && \
    ldconfig && \
    rm -rf /tmp/librtlsdr-build && \
    echo "[✓] krakenrf/librtlsdr built"

# ── 1b. Build KFR DSP library (x86_64 only) ─────────────────────────────────
#        clang++ is required by KFR.
#        git clone is wrapped in an if/else so a network failure (DNS
#        unavailable inside the builder) is non-fatal: the runtime stage
#        already handles a missing libkfr_capi gracefully.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
      if git clone --depth=1 --branch "${KFR_VERSION}" \
             https://github.com/kfrlib/kfr /tmp/kfr 2>&1; then \
        cmake -S /tmp/kfr -B /tmp/kfr/build \
            -DENABLE_CAPI_BUILD=ON \
            -DCMAKE_CXX_COMPILER=clang++ \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_FLAGS="${BUILD_FLAGS}" \
            -DCMAKE_INSTALL_PREFIX=/usr/local && \
        make -C /tmp/kfr/build -j"$(nproc)" && \
        make -C /tmp/kfr/build install 2>&1 | tail -3 && \
        KFR_LIB=$(find /usr/local /tmp/kfr/build -name 'libkfr_capi*.so*' 2>/dev/null | head -1) && \
        if [ -n "$KFR_LIB" ]; then \
          cp "$KFR_LIB" /usr/local/lib/libkfr_capi.so && \
          echo "[✓] libkfr_capi.so installed from real KFR build"; \
        fi && \
        ldconfig && \
        rm -rf /tmp/kfr && \
        echo "[✓] KFR step done"; \
      else \
        echo "[!] KFR: git clone failed (no network in builder) — decimation stub will be used"; \
      fi; \
    fi

# ── 1b-stub. Build minimal kfr_capi stub when real KFR didn't produce the .so ─
#
# KFR 5.x ignores ENABLE_CAPI_BUILD (cmake option renamed); the real .so is
# often not produced.  With decimation_ratio=1 (our default), fir_decimate.c
# takes the  else if (dec == 1)  branch – pure cint8→cfloat32, no FIR math –
# so kfr_filter_process_f32 is NEVER called.  The stub is fully sufficient.
#
# File contents are base64-encoded to avoid Docker heredoc-nesting issues.
#   HDR : kfr/capi.h   (type declarations only – no implementation)
#   STUB: kfr_capi_stub.c  (kfr_allocate→posix_memalign, plan→malloc struct)
RUN if [ "$(uname -m)" = "x86_64" ] && [ ! -f /usr/local/lib/libkfr_capi.so ]; then \
      echo "[*] Building kfr_capi stub (decimation_ratio=1 pass-through)..." && \
      mkdir -p /usr/local/include/kfr && \
      echo 'I3ByYWdtYSBvbmNlCiNpbmNsdWRlIDxzdGRkZWYuaD4KI2lmZGVmIF9fY3BsdXNwbHVzCmV4dGVybiAiQyIgewojZW5kaWYKdHlwZWRlZiBmbG9hdCAga2ZyX2YzMjsKdHlwZWRlZiBkb3VibGUga2ZyX2Y2NDsKdHlwZWRlZiBzdHJ1Y3QgS0ZSX0ZJTFRFUl9GMzIgS0ZSX0ZJTFRFUl9GMzI7CnR5cGVkZWYgc3RydWN0IEtGUl9GSUxURVJfRjY0IEtGUl9GSUxURVJfRjY0Owp2b2lkKiAgICAgICAgICAga2ZyX2FsbG9jYXRlKHNpemVfdCBzaXplKTsKdm9pZCogICAgICAgICAgIGtmcl9yZWFsbG9jYXRlKHZvaWQqIHB0ciwgc2l6ZV90IG5ld19zaXplKTsKdm9pZCAgICAgICAgICAgIGtmcl9kZWFsbG9jYXRlKHZvaWQqIHB0cik7CktGUl9GSUxURVJfRjMyKiBrZnJfZmlsdGVyX2NyZWF0ZV9maXJfcGxhbl9mMzIoY29uc3Qga2ZyX2YzMiogdGFwcywgc2l6ZV90IHRhcF9zaXplKTsKdm9pZCAgICAgICAgICAgIGtmcl9maWx0ZXJfcHJvY2Vzc19mMzIoS0ZSX0ZJTFRFUl9GMzIqIHBsYW4sIGtmcl9mMzIqIG91dCwgY29uc3Qga2ZyX2YzMiogaW4sIHNpemVfdCBzaXplKTsKdm9pZCAgICAgICAgICAgIGtmcl9maWx0ZXJfcmVzZXRfZjMyKEtGUl9GSUxURVJfRjMyKiBwbGFuKTsKdm9pZCAgICAgICAgICAgIGtmcl9maWx0ZXJfZGVsZXRlX3BsYW5fZjMyKEtGUl9GSUxURVJfRjMyKiBwbGFuKTsKI2lmZGVmIF9fY3BsdXNwbHVzCn0KI2VuZGlmCg==' \
        | base64 -d > /usr/local/include/kfr/capi.h && \
      echo 'LyogTWluaW1hbCBLRlIgQ0FQSSBzdHViIOKAkyBzdWZmaWNpZW50IGZvciBkZWNpbWF0aW9uX3JhdGlvPTEgKHBhc3MtdGhyb3VnaCkgKi8KI2luY2x1ZGUgPHN0ZGxpYi5oPgojaW5jbHVkZSA8c3RyaW5nLmg+CnN0cnVjdCBLRlJfRklMVEVSX0YzMiB7IHNpemVfdCBuOyBmbG9hdCogdDsgfTsKdm9pZCoga2ZyX2FsbG9jYXRlKHNpemVfdCBzKSB7CiAgICB2b2lkKiBwID0gTlVMTDsKICAgIGlmIChwb3NpeF9tZW1hbGlnbigmcCwgNjQsIHMgPiAwID8gcyA6IDEpICE9IDApIHJldHVybiBOVUxMOwogICAgcmV0dXJuIHA7Cn0Kdm9pZCoga2ZyX3JlYWxsb2NhdGUodm9pZCogcCwgc2l6ZV90IHMpIHsgZnJlZShwKTsgcmV0dXJuIGtmcl9hbGxvY2F0ZShzKTsgfQp2b2lkICBrZnJfZGVhbGxvY2F0ZSh2b2lkKiBwKSB7IGZyZWUocCk7IH0Kc3RydWN0IEtGUl9GSUxURVJfRjMyKiBrZnJfZmlsdGVyX2NyZWF0ZV9maXJfcGxhbl9mMzIoY29uc3QgZmxvYXQqIHQsIHNpemVfdCBuKSB7CiAgICBzdHJ1Y3QgS0ZSX0ZJTFRFUl9GMzIqIHAgPSBtYWxsb2Moc2l6ZW9mKCpwKSk7CiAgICBpZiAoIXApIHJldHVybiBOVUxMOwogICAgcC0+biA9IG47CiAgICBwLT50ID0gbWFsbG9jKG4gKiBzaXplb2YoZmxvYXQpKTsKICAgIGlmIChwLT50ICYmIHQpIG1lbWNweShwLT50LCB0LCBuICogc2l6ZW9mKGZsb2F0KSk7CiAgICByZXR1cm4gcDsKfQp2b2lkIGtmcl9maWx0ZXJfcHJvY2Vzc19mMzIoc3RydWN0IEtGUl9GSUxURVJfRjMyKiBwLCBmbG9hdCogbywgY29uc3QgZmxvYXQqIGksIHNpemVfdCBuKSB7CiAgICAodm9pZClwOyBtZW1jcHkobywgaSwgbiAqIHNpemVvZihmbG9hdCkpOwp9CnZvaWQga2ZyX2ZpbHRlcl9yZXNldF9mMzIoc3RydWN0IEtGUl9GSUxURVJfRjMyKiBwKSB7ICh2b2lkKXA7IH0Kdm9pZCBrZnJfZmlsdGVyX2RlbGV0ZV9wbGFuX2YzMihzdHJ1Y3QgS0ZSX0ZJTFRFUl9GMzIqIHApIHsKICAgIGlmIChwKSB7IGZyZWUocC0+dCk7IGZyZWUocCk7IH0KfQo=' \
        | base64 -d > /tmp/kfr_capi_stub.c && \
      gcc -O2 -shared -fPIC -o /usr/local/lib/libkfr_capi.so /tmp/kfr_capi_stub.c && \
      ldconfig && \
      echo "[✓] kfr_capi stub built → /usr/local/lib/libkfr_capi.so"; \
    elif [ "$(uname -m)" = "x86_64" ]; then \
      echo "[✓] Real libkfr_capi.so already present, skipping stub"; \
    fi

# ── 1c. Build Heimdall _daq_core ─────────────────────────────────────────────
#        The Makefile default is -O2; we pass CFLAGS to upgrade to -O3.
#        -I/usr/local/include is needed so rtl-sdr.h and kfr/capi.h are found.
# Build all _daq_core targets except decimate (needs kfr_capi/NE10)
# Note: the pre-built librtlsdr.a in _daq_core/ is a specific krakenrf fork
# that includes rtlsdr_set_sample_freq_correction_f and other extensions not
# present in the Firmware/librtlsdr/ submodule. We use it for static linking.
# Step 1: compile shared .o files (sequential – avoids race conditions with -j)
# Step 2: link binaries in parallel
RUN cd /opt/krakensdr/heimdall_daq_fw/Firmware/_daq_core && \
    DAQ_CFLAGS="-Wall -std=gnu99 ${BUILD_FLAGS} -I. -I/usr/local/include" && \
    make daq_util CFLAGS="$DAQ_CFLAGS" && \
    make -j"$(nproc)" rtl_daq rebuffer iq_server CFLAGS="$DAQ_CFLAGS" && \
    echo "[✓] Core DAQ binaries built:" && \
    ls -1 rtl_daq.out rebuffer.out iq_server.out
# Build decimate.out — links against libkfr_capi.so (real or stub)
RUN cd /opt/krakensdr/heimdall_daq_fw/Firmware/_daq_core && \
    if [ "$(uname -m)" = "x86_64" ]; then \
      DAQ_CFLAGS="-Wall -std=gnu99 ${BUILD_FLAGS} -I. -I/usr/local/include" && \
      make decimate_x86 \
           CFLAGS="$DAQ_CFLAGS" \
           LDFLAGS="-L/usr/local/lib" && \
      echo "[✓] decimate.out built" && \
      ls -la decimate.out; \
    else \
      echo "[i] ARM – decimate.out requires NE10, skipped"; \
    fi

# ── 1d. Build gr-krakensdr OOT module ────────────────────────────────────────
RUN cmake -S /opt/krakensdr/gr-krakensdr \
          -B /tmp/gr-build \
          -DCMAKE_INSTALL_PREFIX=/usr \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS="${BUILD_FLAGS}" && \
    make -C /tmp/gr-build -j"$(nproc)" install && \
    ldconfig && \
    rm -rf /tmp/gr-build && \
    echo "[✓] gr-krakensdr installed"


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — runtime
#  Lean image: only the produced artefacts + runtime libraries.
#  All compilers and -dev packages are absent → ~600 MB lighter.
# ════════════════════════════════════════════════════════════════════════════
FROM ubuntu:${UBUNTU_VERSION} AS runtime

ARG GIT_COMMIT=dev
ARG BUILD_FLAGS="-O2 -pipe"

LABEL maintainer="mrheltic"
LABEL description="KrakenSDR + GNU Radio + Heimdall DAQ"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL build.flags="${BUILD_FLAGS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=xcb
ENV QT_X11_NO_MITSHM=1
# Use the GSettings memory backend: no D-Bus session inside the container,
# so dconf would spam warnings. Non-critical settings (window geometry,
# recently opened files) are kept in RAM for the duration of the session.
ENV GSETTINGS_BACKEND=memory
# Force Adwaita icon theme so GRC toolbar and block icons render correctly.
ENV GTK_THEME=Adwaita
# Ensure the krakenrf librtlsdr (in /usr/local/lib) is found before any
# system copy that apt transitive deps might install.
ENV LD_LIBRARY_PATH=/usr/local/lib

# ── 2. Runtime system packages (no *-dev, no compilers) ─────────────────────
RUN apt-get -o Acquire::Retries=5 update && apt-get install -y --no-install-recommends \
    sudo procps iproute2 usbutils lsof \
    netcat-openbsd socat \
    python3 python3-pip \
    python3-numpy python3-scipy python3-matplotlib \
    python3-pyqt5 python3-pyqt5.qtsvg \
    python3-gi python3-gi-cairo gir1.2-gtk-3.0 \
    gnuradio \
    libusb-1.0-0 \
    libzmq5 \
    libportaudio2 \
    pulseaudio pulseaudio-utils libasound2 \
    libgl1 libgl1-mesa-dri \
    x11-apps \
    libxcb1 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-xkb1 libxkbcommon-x11-0 libxkbcommon0 \
    libx11-dev \
    adwaita-icon-theme hicolor-icon-theme \
    gsettings-desktop-schemas \
    fonts-dejavu-core \
    librsvg2-common \
    xterm \
    nano vim \
    && rm -rf /var/lib/apt/lists/*

# Optional GUI editor (gedit + plugins).  Kept in a separate layer with
# '|| true' so a missing or renamed package does not abort the build.
RUN apt-get -o Acquire::Retries=5 update && \
    apt-get install -y --no-install-recommends gedit gedit-plugins 2>/dev/null \
    || apt-get install -y --no-install-recommends gedit 2>/dev/null \
    || echo "[!] gedit not available — use nano/vim instead" ; \
    rm -rf /var/lib/apt/lists/*

# Rebuild icon caches so GTK3 finds Adwaita icons at runtime.
# librsvg2-common provides the gdk-pixbuf SVG loader (its dpkg trigger
# already regenerates the loader cache); we only need to refresh the
# icon theme index files here.
RUN gtk-update-icon-cache -f -t /usr/share/icons/Adwaita \
    && gtk-update-icon-cache -f -t /usr/share/icons/hicolor

# ── 3. Python packages ───────────────────────────────────────────────────────
# One pip call: the resolver honours "numpy<2" across all deps.
# gnuradio from apt is compiled against NumPy 1.x; numpy 2.x breaks it.
RUN pip3 install --no-cache-dir \
    "numpy<2" pyzmq requests pyyaml scikit-rf numba

# ── 4. Copy optimised native libraries from builder ─────────────────────────
# Use RUN --mount to allow graceful handling of optional libkfr_capi
COPY --from=builder /usr/local/lib/librtlsdr.so*  /usr/local/lib/
COPY --from=builder /usr/local/bin/rtl_*          /usr/local/bin/
RUN --mount=type=bind,from=builder,source=/usr/local/lib,target=/tmp/bldr-libs \
    cp /tmp/bldr-libs/libkfr_capi* /usr/local/lib/ 2>/dev/null && \
    echo "[✓] libkfr_capi copied" || echo "[!] libkfr_capi not in builder – decimation disabled"
RUN ldconfig && echo "[✓] Native libraries registered"

# ── 5. Copy gr-krakensdr Python OOT module ──────────────────────────────────
# gr-krakensdr is pure-Python. Install directly from the source tree to avoid
# cmake install path ambiguity (could be python3.10/dist-packages vs
# python3/dist-packages depending on the gnuradio cmake macros version).
RUN --mount=type=bind,from=builder,source=/opt/krakensdr/gr-krakensdr/python/krakensdr,target=/krakensdr-src \
    mkdir -p /usr/lib/python3/dist-packages/gnuradio/krakensdr && \
    cp -r /krakensdr-src/. /usr/lib/python3/dist-packages/gnuradio/krakensdr/ && \
    echo "[✓] gr-krakensdr Python module installed"
# GRC block descriptors (for GNU Radio Companion, optional)
RUN mkdir -p /usr/share/gnuradio/grc/blocks
COPY --from=builder /opt/krakensdr/gr-krakensdr/grc/*.block.yml \
                    /usr/share/gnuradio/grc/blocks/

# ── 6. Copy project tree (Python scripts + compiled Heimdall binaries) ───────
WORKDIR /opt/krakensdr
COPY --from=builder /opt/krakensdr/ ./

# ── 7. udev rules for RTL-SDR ────────────────────────────────────────────────
RUN mkdir -p /etc/udev/rules.d && \
    printf 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE="0666", GROUP="plugdev"\n' \
        > /etc/udev/rules.d/20-rtlsdr.rules && \
    printf 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2832", MODE="0666", GROUP="plugdev"\n' \
        >> /etc/udev/rules.d/20-rtlsdr.rules

# ── 8. Persistent data, logs and external config mount-point ────────────────
# /opt/krakensdr/config is the recommended host-mount target for the
# daq_chain_config.ini override (see docker-compose.yml).
RUN mkdir -p \
    /opt/krakensdr/heimdall_daq_fw/Firmware/_logs \
    /opt/krakensdr/config

# ── 9. Entrypoint ────────────────────────────────────────────────────────────
# Run as root: DAQ processes need chrt -f 99, libusb, rmmod — all require
# root privileges already granted by --privileged (see run.sh).
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /opt/krakensdr
EXPOSE 5000 5001

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
