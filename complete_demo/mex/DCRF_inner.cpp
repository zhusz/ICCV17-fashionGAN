#include "mex.h"
#include "densecrf.h"
#include <cstdio>
#include <cmath>
#include "common.h"
#include <cassert>

/*
// Certainty that the groundtruth is correct
const float GT_PROB = 0.5;

// Simple classifier that is 50% certain that the annotation is correct
MatrixXf computeUnary( const VectorXs & lbl, int M ){
	const float u_energy = -log( 1.0 / M );
	const float n_energy = -log( (1.0 - GT_PROB) / (M-1) );
	const float p_energy = -log( GT_PROB );
	MatrixXf r( M, lbl.rows() );
	r.fill(u_energy);
	//printf("%d %d %d \n",im[0],im[1],im[2]);
	for( int k=0; k<lbl.rows(); k++ ){
		// Set the energy
		if (lbl[k]>=0){
			r.col(k).fill( n_energy );
			r(lbl[k],k) = p_energy;
		}
	}
	return r;
}
*/

// new_map = DCRF(image, prob, M, H, W, params)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int M;
    int H;
    int W;
    if (nrhs == 6)
    {
        M = mxGetScalar(prhs[2]);
        H = mxGetScalar(prhs[3]);
        W = mxGetScalar(prhs[4]);
    }
    else
    {
        mexErrMsgTxt("Only six inputs are allowed: new_map = DCRF(image, prob, M, H, W, params)");
    }

    if (nlhs > 1)
    {
        mexErrMsgTxt("At most one output is allowed: new_map = DCRF(image, prob, M)");
    }
    if (!mxIsUint8(prhs[0])) mexErrMsgTxt("image must be uint8!");
    if (!mxIsDouble(prhs[1])) mexErrMsgTxt("map must be uint8!");
    unsigned char * im = (unsigned char*)mxGetPr(prhs[0]);
    // mexPrintf("%d %d %d %d", im[0], im[1], im[2], im[3]);
    double * prob = mxGetPr(prhs[1]);
    double * params = mxGetPr(prhs[5]);

    // ---------------------- Use the original demo code ----------------------- //
    MatrixXf unary(M, H * W);
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            for (int k = 0; k < M; k++)
            {
                unary(k,i+j*H) = -log(prob[k+i*M+j*M*H]);
            }
        }
    }
    // for (int i = 0; i < 10; i++) mexPrintf("%d ", (int)prob[i]);
    // mexPrintf("\n");
    // for (int i = 0; i < 10; i++) mexPrintf("%d ", im[i]);
    // return;

//	MatrixXf unary = computeUnary( getLabeling( anno, W*H, M ), M );
	DenseCRF2D crf(W, H, M);
	crf.setUnaryEnergy( unary );
	// crf.addPairwiseGaussian( 3, 3, new PottsCompatibility( 3 ) );
	// crf.addPairwiseBilateral( 80, 80, 13, 13, 13, im, new PottsCompatibility( 10 ) );
	// VectorXs map = crf.map(5);
    crf.addPairwiseGaussian(params[0], params[1], new PottsCompatibility(params[2]));
    crf.addPairwiseBilateral(params[3], params[4], params[5], params[6], params[7], im, new PottsCompatibility(params[8]));
    VectorXs map = crf.map(params[9]);

	// unsigned char *res = colorize( map, W, H);
    plhs[0] = mxCreateDoubleMatrix(H*W, 1, mxREAL);
    double *output = mxGetPr(plhs[0]);
    for (int i = 0; i < W*H; i++) output[i] = (double)map[i];

    return;
}
