#include "myrand.h"
#include "particles.h"
#include "physics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

//ldoc
/**
 * # Main driver
 *
 * The simulation takes a list of physical parameters, an initial
 * condition, a number of particles, and some information about
 * the time stepping and the rate of frame output for visualization.
 */

void simulate(sim_param_t* params,   // Physics parameters
              const char* ic_name,   // IC name (ric or circle)
              int N,                 // Number of particles
              int rendert,           // Time steps between frame dumps
              float tfinal,          // End time
              float dt)              // Time step
{
    //#pragma omp parallel
    //{
    float radius     = params->radius;
    float L          = 1.03*sqrtf(M_PI*radius*radius*N);
    int   nbinx      = (int) (L/(4*radius));

    particles_t* particles = alloc_particles_t(nbinx, N, L);
    cell_t* cells = alloc_cells(nbinx, N);
    if (strcmp(ic_name, "ric") == 0)
        init_ric(particles, params->vhappy);
    else
        init_circle(particles, params->vhappy);

    FILE* fp = start_frames("particles.csv");
    int frames = 0;
    int relocate_time = 1;
    for (float t=0.0; t < tfinal; t += dt) {
        if (rendert > 0 && frames % rendert == 0)
            write_frame(fp, particles);
        if (relocate_time % 10 == 0) {
            copy_cells(cells, particles);
        }
        else {
            compute_nbr_lists(particles);
        }
        compute_forces(particles, params);
        leapfrog_step(particles, dt);
        ++frames;
        ++relocate_time;
    }
    end_frames(fp);
    free_particles_t(particles);
    free_cells(cells, nbinx);
    //}
}

/**
 * Our `main` function uses the `getopt` library to parse options,
 * then runs the simulation.  This is the type of thing that could
 * easily be done by access to a little language for reading configuration
 * files as well -- I like Lua for this purpose.
 */

int main(int argc, char **argv)
{
    double ostart, oend;
    ostart = omp_get_wtime();

    sim_param_t params;
    set_default_params(&params);
    const char* ic_name = "circle";
    
    int seed     = 0;      // Random seed
    int N        = 1000;   // Number of particles
    int rendert  = 0;      // Time between dumped frames
    float tfinal = 1e3;    // Final time
    float dt = 1e-1;       // Time step

    // Argument processing
    int c;
    extern char* optarg;
    while ((c = getopt(argc, argv, "?r:e:v:a:s:d:N:T:h:f:")) != -1) {
        switch (c) {
        case '?':
            fprintf(stderr,
                    "%s\n"
                    "\t-?: print this message\n"
                    "\t-i: initial conditions (%s)\n"
                    "\t-r: interaction radius (%g)\n"
                    "\t-e: interaction strength epsilon (%g)\n"
                    "\t-v: preferred speed for moshers (%g)\n"
                    "\t-a: flocking force multiplier (%g)\n"
                    "\t-s: noise variance (%g)\n"
                    "\t-d: damping force multiplier (%g)\n"
                    "\t-N: number of particles (%d)\n"
                    "\t-T: final time (%g)\n"
                    "\t-h: time step (%g)\n"
                    "\t-f: steps between output frames (%d)\n",
                    argv[0], ic_name,
                    params.radius, params.epsilon, params.vhappy,
                    params.alpha, params.sigma, params.damp,
                    N, tfinal, dt, rendert);
            return -1;
        case 'i': ic_name        = optarg;       break;
        case 'r': params.radius  = atof(optarg); break;
        case 'e': params.epsilon = atof(optarg); break;
        case 'v': params.vhappy  = atof(optarg); break;
        case 'a': params.alpha   = atof(optarg); break;
        case 's': params.sigma   = atof(optarg); break;
        case 'd': params.damp    = atof(optarg); break;
        case 'N': N              = atoi(optarg); break;
        case 'T': tfinal         = atof(optarg); break;
        case 'h': dt             = atof(optarg); break;
        case 'f': rendert        = atoi(optarg); break;
        default:
            fprintf(stderr, "Unknown option (-%c)\n", c);
            return -1;
        }
    }

    ran_seed(seed);
    simulate(&params, ic_name, N, rendert, tfinal, dt);

    oend = omp_get_wtime();
    printf("Total Time: %f\n", oend-ostart);

    return 0;
}

