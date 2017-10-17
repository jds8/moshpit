#include "particles.h"
#include "myrand.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//ldoc
/**
 * # Particle data stucture implementation
 *
 * The particle data structure is one of the places where you may
 * want to improve the implementation.  There are several aspects
 * that are not entirely satisfactory; are we better off with the
 * parallel array layout we have now (which is good for vectorization,
 * but perhaps not locality), or with an array-of-structs layout?
 * Should the particle type (which is really only a single bit of
 * information) be stored as a full integer, or as some smaller data
 * type?  Is there a better way of handling the bin data structures
 * (noting that this is not the bin data structure from the original
 * code)?  Play with things and find out!
 * 
 * ## Allocate and free particle data
 */

particles_t* alloc_particles_t(int nbinx, int N, float L)
{
    int size_total = nbinx*nbinx;
    particles_t* p = malloc(sizeof(particles_t));
    p->nbinx = nbinx;
    p->N = N;
    p->L = L;
    p->particles = (particle_t*) calloc(N, sizeof(particle_t));
    p->bins = (particle_t**) calloc(size_total, sizeof(particle_t*));
    return p;
}

void free_particles_t(particles_t* p)
{
    free(p->particles);
    free(p->bins);
    free(p);
}


/**
 * ## Neighbor list computations
 *
 * Right now, computing a neighbor list means computing the `cells`
 * and `next` arrays; we don't touch the position, velocity, or force
 * arrays.  But perhaps we should!  As currently written, traversing
 * a neighbor list involves jumping all over memory to grab the information
 * for different particles; you may want to consider something that makes
 * better use of cache locality.
 */

void compute_nbr_lists(particles_t* particles)
{
    particle_t* restrict part = particles->particles;
    particle_t** restrict bins = particles->bins;
    float L = particles->L;
    int N = particles->N;
    int nbinx = particles->nbinx;

    // Recompute neighbor list
    /*const int size_total = nbinx*nbinx;
    memset(bins, 0, size_total * sizeof(int));*/
    
    #pragma omp simd
    for (int i=0; i<nbinx*nbinx; i++) {
        bins[i] = NULL;
    }
    
    for (int i=0; i<N; i++) {
        int binx = coord_to_index(part[i].xx, nbinx, L);
        int biny = coord_to_index(part[i].xy, nbinx, L);
        int b = binx + biny*nbinx;
        part[i].next = bins[b];
        bins[b] = &(part[i]);
    }
}

/**
 * ## Initialization
 *
 * The `initial_circle` function puts all the active moshers in a big
 * circle in the middle of the domain at the start; the `init_ric`
 * function distributes the active moshers uniformly through the
 * domain and assigns them a random velocity as well as position.
 * It might be interesting to see how the behavior depends on this
 * initial distribution.
 */


void init_ric(particles_t* particles, float speed)
{
    particle_t* part = particles->particles;
    int N = particles->N;
    float L = particles->L;

    //#pragma omp for
    for (int i=0; i<N; i++) {
        float t = 2*M_PI*ran_ran2();

        part[i].xx = L*ran_ran2();
        part[i].xy = L*ran_ran2();

        if (ran_ran2() > 0.16){
            part[i].vx = 0.0;
            part[i].vy = 0.0;
            part[i].type = BLACK;
        } else {
            part[i].vx = speed * sin(t);
            part[i].vy = speed * cos(t);
            part[i].type = RED;
        }
    }
}


void init_circle(particles_t* particles, float speed)
{
    particle_t* part = particles->particles;
    int N = particles->N;
    float L = particles->L;
    
    //#pragma omp for
    for (int i=0; i<N; i++){
        float tx = L*ran_ran2();
        float ty = L*ran_ran2();
        float tt = 2*M_PI*ran_ran2();

        part[i].xx = tx;
        part[i].xy = ty;

        // the radius for which 30% of the particles are red on avg
        float dd2 = (tx-L/2)*(tx-L/2) + (ty-L/2)*(ty-L/2);
        float rad2 = 0.16*L*L / M_PI;

        if (dd2 < rad2)
            part[i].type = RED;
        else
            part[i].type = BLACK;

        if (part[i].type == RED) {
            part[i].vx = speed*cos(tt);
            part[i].vy = speed*sin(tt);
        } else {
            part[i].vx = 0.0;
            part[i].vy = 0.0;
        }
    }
}


/**
 * ## I/O subsystem
 * 
 * The I/O subsystem writes out a text file with particle information;
 * given half a chance, this takes more time than just about anything else
 * in the simulation!  We could be somewhat more space efficient using
 * a binary format (and you are welcome to do so, though the visualizer
 * will need corresponding changes) -- but really, the "right" approach
 * is probably to moderate our desire to dump out too much information.
 * The pictures are pretty, but summary statistics are what the researchers
 * who designed this simulation were really after.
 */


FILE* start_frames(const char* fname)
{
    FILE* fp = fopen(fname, "w");
    if (fp == NULL) {
        fprintf(stderr, "Could not open %s for output\n", fname);
        exit(-1);
    }
    fprintf(fp, "PTag,PId,PLocX,PLocY,PDirX,PDirY\n");
    return fp;
}


void write_frame(FILE* fp, particles_t* particles)
{
    particle_t* part = particles->particles;
    int n = particles->N;
    float L = particles->L;
    for (int i = 0; i < n; ++i) {
        fprintf(fp, "%d,%d,%g,%g,%g,%g\n", part[i].type, i+1,
                part[i].xx/L, part[i].xy/L,
                part[i].vx/L, part[i].vy/L);
    }
}


void end_frames(FILE* fp)
{
    fclose(fp);
}

