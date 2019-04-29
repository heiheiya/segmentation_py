#include <iostream>
#include <Python.h>
#include <windows.h>

using namespace std;

int predictImage(char* image, char* pb, int cropHeight, int cropWidth, char* dataset)
{
	try
	{
		Py_Initialize();
		PyEval_InitThreads();
		PyObject* pFunc = NULL;
		PyObject* pArg = PyTuple_New(5);
		PyObject* module = NULL;

		PyRun_SimpleString("import sys");
		PyRun_SimpleString("sys.path.append('./')");

		module = PyImport_ImportModule("predict_pb");
		if (!module)
		{
			cout << "ERROR: Cannot open module..." << endl;
			PyErr_Print();
			Py_Finalize();
			return -1;
		}

		pFunc = PyObject_GetAttrString(module, "predictFromPb");
		if (!pFunc)
		{
			cout << "ERROR: Cannot open function..." << endl;
			Py_Finalize();
			return -1;
		}

		PyTuple_SetItem(pArg, 0, Py_BuildValue("s", image));
		PyTuple_SetItem(pArg, 1, Py_BuildValue("s", pb));
		PyTuple_SetItem(pArg, 2, Py_BuildValue("i", cropHeight));
		PyTuple_SetItem(pArg, 3, Py_BuildValue("i", cropWidth));
		PyTuple_SetItem(pArg, 4, Py_BuildValue("s", dataset));
		if (module != NULL)
		{
			PyGILState_STATE gstate;
			gstate = PyGILState_Ensure();
			PyObject* pRet = PyObject_CallObject(pFunc, pArg);
			PyGILState_Release(gstate);
			if (!pRet)
			{
				cout << "ERROR: Call object failed..." << endl;
				Py_Finalize();
				return -1;
			}			
		}
		return 0;

	}
	catch (exception* e)
	{
		cout << "Standard exception: " << e->what() << endl;
	}
}

int main()
{
	char* imagePath = "image/0001TP_007170.png";
	char* pbPath = "model/AdapNet_frozen_graph_meta.pb";
	int cropHeight = 512;
	int cropWidth = 512;
	char* dataset = "CamVid";

	predictImage(imagePath, pbPath, cropHeight, cropWidth, dataset);

	system("pause");
	return 0;
}