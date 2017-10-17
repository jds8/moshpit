#include "physics.h"
#include "particles.h"
#include "myrand.h"

#include <string.h>
#include <float.h>
#include <math.h>

/**
 * # Physics implementation
 *
 * This is the module where most of your time will be spent.
 * The `compute_forces` routine, in particular, takes an enormous
 * fraction of the total run time.  Your job is to speed it up,
 * by serial tuning or by parallelism!
 *
 * ## Default parameter initialization
 */

void set_default_params(sim_param_t* p)
{
    p->radius  = 1.0;
    p->epsilon = 25.0;
    p->vhappy  = 1.0;
    p->alpha   = 0.9;
    p->sigma   = 0.1;
    p->damp    = 1.0;
}


/**
 * ## Helper functions
 *
 * The `mymod` and `mod_rvec` functions are there to handle the
 * modular arithmetic for the wrap-around boundary conditions.
 * The function `mymod(a,b)` just computes the remainder of `a`
 * on division by `b`.  The `mod_rvec` function is more interesting;
 * it returns integer `a` modulo `b`, and sets the `image` flag if
 * this actually involves returning something other than `a`.
 */


inline float mymod(float a, float b)
{
  return a - b*(int)(a/b) + b*(a<0);
}


inline int mod_rvec(int a, int b, int *image)
{
    *image = 1;
    if (a>=b)  return a-b;
    if (a< 0)  return a+b;
    *image = 0;
    return a;
}


/**
 * ## Force computation function
 * 
 * The force computation function is the home to most of the interesting
 * code in the project.  We loop through particles, and compute the
 * interactions of each with other nearby particles.  Then we sum all
 * the forces and move on.  But most of the time in this code is spent
 * not in force computations, but rather in fetching data from memory.
 * A traversal that processes particles one bucket at a time -- rather
 * than in the particle identifier order -- will probably have better
 * memory locality.  I suggest you try it!  I have done enough preliminary
 * tuning that I suspect you will not make significant improvements to
 * the performance unless you first address the issues with the memory
 * access pattern.
 */

void compute_forces(particles_t* particles, sim_param_t* params)
{
    const float radius     = params->radius;
    const float epsilon    = params->epsilon;
    const float vhappy_red = params->vhappy;
    const float alpha      = params->alpha;
    const float sigma      = params->sigma;
    const float damp_coeff = params->damp;

    const float R   = 2*radius;
    const float R2  = R*R;
    const float FR  = 2*R;
    const float FR2 = FR*FR;

    particle_t** restrict bins = particles->bins;
    particle_t* restrict part = particles->particles;

    int nbinx = particles->nbinx;
    float L = particles->L;

    //#pragma omp for
    for (int i=0; i<nbinx*nbinx; i++) {
        for (particle_t* pt=bins[i]; pt!=NULL; pt=pt->next) {
            float fx = 0.0;
            float fy = 0.0;
            float wx = 0.0;
            float wy = 0.0;
            float xi = part[i].xx;
            float yi = part[i].xy;
            int is_redi = (part[i].type == RED);

            int binx = coord_to_index(xi, nbinx, L);
            int biny = coord_to_index(yi, nbinx, L);

            for (int ix=-1; ix<=1; ++ix) {
            for (int iy=-1; iy<=1; ++iy) {

                int imagex, imagey;
                int jx = mod_rvec(binx+ix,nbinx,&imagex);
                int jy = mod_rvec(biny+iy,nbinx,&imagey);

                int ind = jx + jy*nbinx;
                float xic = xi - (!imagex ? 0.0 : L*ix);
                float yic = yi - (!imagey ? 0.0 : L*iy);

                for (particle_t* nbr=bins[ind]; nbr!=NULL; nbr=nbr->next) {
                    // Distance to neighbor n
                    float dx = nbr->xx - xic;
                    float dy = nbr->xy - yic;
                    float dist = dx*dx + dy*dy;

                    if (dist > 1e-10) {
                        //===============================================
                        // force calculation - hertz
                        if (dist < R2) {
                            float l   = sqrtf(dist);
                            float co1 = (1-l/R);
                            float co  = epsilon * co1*sqrtf(co1);
                            float c   = co/l;
                            fx -= c * dx;
                            fy -= c * dy;
                        }

                        //===============================================
                        // add up the neighbor velocities
                        if (is_redi && nbr->type == RED && dist < FR2) {
                            wx += part[i].vx;
                            wy += part[i].vy;
                        }
                    }
                    //neighbor = neighbor->next;
                }
            } }

            //=====================================
            // flocking force
            float wlen2 = wx*wx + wy*wy;
            if (is_redi && wlen2 > 1e-12) {
                float c = alpha / sqrtf(wlen2);
                fx += c * wx;
                fy += c * wy;
            }

            //====================================
            // self-propulsion
            float vx = part[i].vx;
            float vy = part[i].vy;
            float vlen2 = vx*vx + vy*vy;
            if (vlen2 > 1e-12) {
                float vhappy = is_redi * vhappy_red;
                float vlen = sqrtf(vlen2);
                float c = damp_coeff*(vhappy-vlen)/vlen;
                fx += c * vx;
                fy += c * vy;
            }

            //=======================================
            // noise term
            if (is_redi) {
                // Box-Muller method
                float u1 = ran_ran2();
                float u2 = 2*M_PI*ran_ran2();
                float lfac = sqrtf(-2*log(u1));
                fx += sigma*lfac*cos(u2);
                fy += sigma*lfac*sin(u2);
            }

            part[i].fx = fx;
            part[i].fy = fy;
        }
    }
}


/**
 * ## Force computation function
 * 
 * The Newton-Stormer-Verlet (or "leapfrog") algorithm is a classic
 * time-stepping algorithm that updates velocities based on forces
 * and positions based on velocities.  Technically, the "natural" times
 * associated with the positions and velocities are separated by half
 * a time step (hence the name "leapfrog").
 *
 * In the inner loop, we also deal with the periodic boundary condition,
 * "wrapping around" the positions of any particles that move over the domain
 * boundary as the result of the time step.
 */

void leapfrog_step(particles_t* restrict particles, float dt)
{
    particle_t* restrict part = particles->particles;
    int N = particles->N;
    float L = particles->L;

    // Integrate the forces
    //#pragma omp for
    for (int i=0; i<N;i++) {
        // Newton-Stomer-Verlet
        part[i].vx += part[i].fx * dt;
        part[i].vy += part[i].fy * dt;

        part[i].xx += part[i].vx * dt;
        part[i].xy += part[i].vy * dt;

        // boundary conditions
        if (part[i].xx >= L*(1-FLT_EPSILON) || part[i].xx < FLT_EPSILON)
            part[i].xx = mymod(part[i].xx, L);
        if (part[i].xy >= L*(1-FLT_EPSILON) || part[i].xy < FLT_EPSILON)
            part[i].xy = mymod(part[i].xy, L);

    }
}
