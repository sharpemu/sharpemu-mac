// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// No guest code or GPU. Reproduce the fixed mapping and first libc write in
// the ASTRO BOT boot log, using either of the macOS provider's backing types.
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    const uint64_t bytes = 0x7e000000;
    const uint64_t object_bytes = 2 * 1024 * 1024;
    const uint64_t extent = 0x80000000;
    mach_port_t task = mach_task_self();
    mach_vm_address_t destination = 0x300000000, alias = 0x10000000000;
    // Relocated linker metadata places the host stack above the guest application's
    // limit; the PIE flag is retained. A fixed reservation must never overwrite
    // an existing host mapping, including an active stack or its guard pages.
    if ((uintptr_t)&task < 0xfc00000000) return 1;
    if (mach_vm_allocate(task, &destination, extent, VM_FLAGS_FIXED) != KERN_SUCCESS) return 2;
    if (destination != 0x300000000 || mprotect((void *)destination, extent, PROT_NONE)) return 3;
    if (argc == 2 && !strcmp(argv[1], "shm")) {
        char name[64];
        snprintf(name, sizeof(name), "/sharpemu-map-%d", getpid());
        int descriptor = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
        if (descriptor < 0) return 4;
        shm_unlink(name);
        if (ftruncate(descriptor, extent)) return 5;
        void *view = mmap((void *)destination, bytes, PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_FIXED, descriptor, 0);
        alias = (mach_vm_address_t)mmap((void *)alias, extent, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
        close(descriptor);
        if (view == MAP_FAILED || alias == (mach_vm_address_t)MAP_FAILED) return 6;
    } else {
        if (mach_vm_map(task, &alias, extent, 0xffff, VM_FLAGS_ANYWHERE, 0, 0, FALSE,
                VM_PROT_NONE, VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_SHARE) != KERN_SUCCESS) return 7;
        for (uint64_t offset = 0; offset < extent; offset += object_bytes) {
            memory_object_size_t size = object_bytes;
            mach_port_t entry = 0;
            if (mach_make_memory_entry_64(task, &size, 0,
                    MAP_MEM_NAMED_CREATE | MAP_MEM_PURGABLE | VM_PROT_READ | VM_PROT_WRITE,
                    &entry, 0) != KERN_SUCCESS) return 8;
            mach_vm_address_t target = alias + offset;
            kern_return_t mapped = mach_vm_map(task, &target, object_bytes, 0,
                VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, entry, 0, FALSE,
                VM_PROT_READ | VM_PROT_WRITE, VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_SHARE);
            mach_port_deallocate(task, entry);
            if (mapped != KERN_SUCCESS || target != alias + offset) return 9;
        }
        vm_prot_t current = 0, maximum = 0;
        mach_vm_address_t target = destination;
        if (mach_vm_remap(task, &target, bytes, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
                task, alias, FALSE, &current, &maximum, VM_INHERIT_SHARE) != KERN_SUCCESS) return 10;
        if (target != destination || mprotect((void *)destination, bytes, PROT_READ | PROT_WRITE)) return 11;
    }
    if (alias < 0x10000000000) return 16;
    const uint64_t offsets[] = { 0x20, object_bytes - 1, object_bytes, bytes - 1 };
    for (unsigned int index = 0; index < sizeof(offsets) / sizeof(offsets[0]); index++) {
        ((volatile unsigned char *)destination)[offsets[index]] = 0x51 + index;
        if (((volatile unsigned char *)alias)[offsets[index]] != 0x51 + index) return 12;
        ((volatile unsigned char *)alias)[offsets[index]] = 0xa1 + index;
        if (((volatile unsigned char *)destination)[offsets[index]] != 0xa1 + index) return 13;
    }
    printf("fixed-guest-map-pass backing=%s stack=0x%llx guest=0x%llx size=0x%llx\n",
        argc == 2 ? argv[1] : "chunked", (unsigned long long)(uintptr_t)&task,
        (unsigned long long)destination, (unsigned long long)bytes);
    if (mach_vm_deallocate(task, alias, extent) != KERN_SUCCESS) return 14;
    return mach_vm_deallocate(task, destination, extent) == KERN_SUCCESS ? 0 : 15;
}
