https://www.hackster.io/mabushaqraedu0/axi-lite-slave-custom-ip-core-to-control-led-under-petalinux-b70b1d

TODO: verilator
https://www.chisel-lang.org/docs/installation#java-versions

```
curl -O "https://github.com/sbt/sbt/releases/download/v1.11.1/sbt-1.11.1.zip"
curl -O "https://download.oracle.com/java/17/archive/jdk-17.0.12_windows-x64_bin.zip"
```

setup_env.sh
```
@echo off
set "PATH=%installdir%\Programs\sbt\bin;%PATH%"
REM https://stackoverflow.com/a/76151135
set "PATH=%installdir%\Programs\Java\jdk-17.0.12\bin;%PATH%"
```
