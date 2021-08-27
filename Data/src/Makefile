#
# RAMS 
# Range Dependence Acoustic software
# base on RAM1.5 , for the environment with solid seabed
#

#Find all source files, create a list of corresponding object files
SRCS:=$(wildcard *.f)
OBJS:=$(patsubst %.f,%.o,$(SRCS))

## Ditto for mods (They will be in both lists)
MODS:=$(wildcard *.f)
MOD_OBJS:=$(patsubst %.f,%.o,$(MODS))

# Compiler/Linker settings
FC := gfortran
FLFLAGS := -w -Os
FCFLAGS := -w -Os 

PROGRAM := ../bin/ram
PRG_OBJ := $(PROGRAM).o

# make without parameters will make first target found.
default := $(PROGRAM)

# Compiler steps for all objects

%.o: %.for
	$(FC) $(FCFLAGS) -c $< -o $@
	
# Linker
$(PROGRAM) : $(OBJS)
	$(FC) $(FLFLAGS) -o $@ $^

# If something doesn't work right, have a 'make debug' to 
# show what each variable contains.
debug:
	@echo "SRCS = $(SRCS)"
	@echo "OBJS = $(OBJS)"
	@echo "MODS = $(MODS)"
	@echo "MOD_OBJS = $(MOD_OBJS)"
	@echo "PROGRAM = $(PROGRAM)"
	@echo "PRG_OBJ = $(PRG_OBJ)"

clean:
	rm -rf $(OBJS) $(PROGRAM) $(%.o,%.mod,$(MOD_OBJS))

.PHONY: debug default clean

