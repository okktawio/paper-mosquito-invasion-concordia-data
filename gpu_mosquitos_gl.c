float expit(float p)
{
  return 1.00 / (1.00 + native_exp(-p));
}


float logit(float p)
{
  float casiuno = 1 - 1e-9;
  
  if(p > casiuno) p = casiuno;
  if(p < 1e-9) p = 1e-9;
  return log(p/(1 - p));
}

kernel void init_mapa(int X, int Y,
		      float Kbaldio,
		      global float * K,
		      global int   * M,
		      global float * U)
{
  uint x = get_global_size(0);
  uint i = get_global_id(0);
  uint j = get_global_id(1);
  if(i < X && j < Y)
    {
      float m       = (float) M[i * Y + j];
      K [i * Y + j] = (float)((m > 0) + Kbaldio * (m == 0) + 0.0001 * (m < 0));
      U [i * Y + j] = 0;
    } 
}

kernel void siembra(int X, int Y, float x, float y, float N, global float * U)
{
  uint ii = get_global_id(0);
  uint jj = get_global_id(1);
  
  if(ii < X && jj < Y)
    {
      float x0 = floor(x);
      float y0 = floor(y);
      float xr = 1 - (x - x0);
      float yr = 1 - (y - y0);
      int xi = (int) x0;
      int yi = (int) y0;
      U[ii * Y + jj] += 0.001 * ((ii == yi)       * (jj == xi))       * (xr * yr * N) ;
      U[ii * Y + jj] += 0.001 * ((ii == (yi + 1)) * (jj == xi))       * (xr * (1 - yr) * N) ;
      U[ii * Y + jj] += 0.001 * ((ii == yi)       * (jj == (xi + 1))) * ((1 - xr) * yr * N) ;
      U[ii * Y + jj] += 0.001 * ((ii == (yi + 1)) * (jj == (xi + 1))) * ((1 - xr) * (1 - yr) * N) ;

    }
}

kernel void rdmosquitos(int X, int Y, int T,
			global float * params,
			global float * clima,
			global float * U,
			global float * K,
			global int   * Mc)
{
  uint x = get_global_size(0);
  uint ii = get_global_id(1);
  uint jj = get_global_id(0);
  
  float oo = params[0];
  float pe = params[1];
  float mxr = params[2];
  float per = params[3];
  float r0  = params[4];
  float v   = params[9];
  
  float kc = expit(oo + pe * clima[T]);
  
  float r  = r0;
  if(per > 0)
    r = mxr * expit(logit(r0 / mxr) + per * clima[T]);
  
  float D = params[5];
  float Dbaldio = params[6];
  float dx = params[7];
  float dt = params[8];
  
  
  
  if(ii < Y && jj < X)
    {
      
      float Kt = K[jj * Y + ii] * kc;
      //int ii = i%j;
      //int jj = i/j;
      float top = 0;
      if(jj > 0) top = U[(jj - 1) * Y + ii];
      float bottom = 0;
      if(jj < (X - 1)) bottom = U[(jj + 1) * Y + ii];
      float left = 0;
      if(ii > 0) left = U[jj * Y + ii - 1];
      float right = 0;
      if(ii < Y - 1) right = U[jj * Y + ii + 1];
      float deltaU = (top + bottom + left + right - 4 * U[jj * Y + ii]) / (dx * dx);
      float difusion = (D * (Mc[jj * Y + ii] > 0) + Dbaldio * (Mc[jj * Y + ii] == 0));
      if (fabs(v) < 1e-7) v = 1e-7;
      float logistico = 0;
      float U0 = U[jj * Y + ii];
      if(U[jj * Y + ii] > 0)
	logistico = r * U[jj * Y + ii] * (1. - native_powr(U[jj * Y + ii] / Kt, v )) / v;
      /*
      if(ii == 160 && jj == 40 && T > 295 && T < 305) 
	printf("ii %3d jj %3d T %3d r0 %f r %f v %f U %f K %f div %f logistico %f pot %f -> U %f\n",
	       ii, jj, T, v, r0, r,
	       U0,
	       Kt,
	       U0 / Kt,
	       logistico,  native_powr(U0 / Kt, v ),
	       U[jj * Y + ii]);
      */
      U[jj * Y + ii] = U[jj * Y + ii] + dt * (difusion * deltaU + logistico);
      if(isnan(U[jj * Y + ii])) U[jj * Y + ii] = 0;
      if(isinf(U[jj * Y + ii])) U[jj * Y + ii] = 0;
    }
}


kernel void get_esperados(int X, int Y, int T,
			  global float * U,
			  global int   * mosquitos,
			  global float * esperados,
			  uint N)
{
  uint kk = get_local_id(2);
  uint i = get_group_id(0);
  uint j = get_group_id(1);
  uint K = get_local_size(2);
  uint k = 0;
  for(uint ii = 0; ii < N; ii += K)
    {
      k = ii + kk;
      if(k < N)
	{
	  uint y = mosquitos[k * 6];
	  uint x = mosquitos[k * 6 + 1];
	  uint t = mosquitos[k * 6 + 2];
	  if(x == i && y == j && t == T)
	    esperados[k] = U[x * Y + j];
	}
    }
}

