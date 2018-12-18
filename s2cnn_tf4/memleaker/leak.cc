#include <stdio.h>

int slowfib(int n)
{
	if (n <= 1)
	{ 
		return 1;
	}
	else
	{
		return slowfib(n-1) + slowfib(n-2);
	}
}

int main(int argc, char** argv)
{
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 1024*1024; j++)
		{
			char *leaker = new char[1024];
		}
		printf("leaked %dGB slowfib: %d\n",i+1, slowfib(35));
	}
	return 0;
}
