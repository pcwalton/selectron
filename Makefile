HSA_DEVEL_ROOT=\Users\pcwalton\Applications\HSA-Devel-Beta\sdk
CFLAGS=/Zi

all:	selectron-cl

selectron-cl.exe:	selectron-cl.cpp selectron.h
	cl $(CFLAGS) /I$(HSA_DEVEL_ROOT)\include selectron-cl.cpp /link $(HSA_DEVEL_ROOT)\lib\x86\OpenCL.lib

selectron-cl:	selectron-cl.cpp selectron.h
	g++ -O2 -DNO_SVM -o selectron-cl -framework OpenCL selectron-cl.cpp

