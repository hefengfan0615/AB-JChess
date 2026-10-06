# AB-JChess engine build entry point

.DEFAULT_GOAL := build

# Preserve user flags before entering the source Makefile recursively.
ifeq ($(MAKELEVEL),0)
export ENV_CXXFLAGS := $(CXXFLAGS)
export ENV_DEPENDFLAGS := $(DEPENDFLAGS)
export ENV_LDFLAGS := $(LDFLAGS)
endif

.PHONY: all build help profile-build strip install clean objclean profileclean \
	net analyze format config-sanity abjnnue-test abjnnue-sha256-test \
	position-protocol-test skyrule-jieqi-test

all: build

build help profile-build strip install clean objclean profileclean net analyze \
format config-sanity abjnnue-test abjnnue-sha256-test position-protocol-test \
skyrule-jieqi-test:
	$(MAKE) -C src $@
