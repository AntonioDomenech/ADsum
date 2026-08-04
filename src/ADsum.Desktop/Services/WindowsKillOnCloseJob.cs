using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace ADsum.Desktop.Services;

/// <summary>
/// Windows Job Object that contains a worker and its descendants. Closing or
/// terminating the job kills every contained process, including children that
/// inherited redirected standard handles.
/// </summary>
internal sealed class WindowsKillOnCloseJob : IDisposable
{
    private const uint JobObjectLimitKillOnJobClose = 0x00002000;
    private const uint Th32csSnapProcess = 0x00000002;
    private const int ErrorNoMoreFiles = 18;
    private const int ErrorInvalidParameter = 87;
    private readonly SafeJobHandle _handle;
    private int _disposed;

    private WindowsKillOnCloseJob(SafeJobHandle handle)
    {
        _handle = handle;
    }

    public static WindowsKillOnCloseJob Create()
    {
        var handle = CreateJobObject(IntPtr.Zero, null);
        if (handle.IsInvalid)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error(), "Unable to create the local speech Windows Job Object.");
        }

        var information = new JobObjectExtendedLimitInformation
        {
            BasicLimitInformation = new JobObjectBasicLimitInformation
            {
                LimitFlags = JobObjectLimitKillOnJobClose
            }
        };
        var size = Marshal.SizeOf<JobObjectExtendedLimitInformation>();
        var pointer = Marshal.AllocHGlobal(size);
        try
        {
            Marshal.StructureToPtr(information, pointer, false);
            if (!SetInformationJobObject(handle, JobObjectInfoType.ExtendedLimitInformation, pointer, (uint)size))
            {
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "Unable to configure the local speech Windows Job Object.");
            }
        }
        catch
        {
            handle.Dispose();
            throw;
        }
        finally
        {
            Marshal.FreeHGlobal(pointer);
        }

        return new WindowsKillOnCloseJob(handle);
    }

    /// <summary>
    /// Assigns the launched process and any children it created between
    /// Process.Start and this call. Once the root is assigned, later children
    /// inherit the job automatically.
    /// </summary>
    public void AssignProcessTree(int rootProcessId)
    {
        ThrowIfDisposed();
        var previouslyObserved = new HashSet<int>();
        var stablePasses = 0;
        for (var pass = 0; pass < 8; pass++)
        {
            var processIds = DescendantProcessIds(rootProcessId);
            var foundNewProcess = false;
            foreach (var processId in processIds)
            {
                if (previouslyObserved.Add(processId))
                {
                    foundNewProcess = true;
                    AssignProcess(processId);
                }
            }

            stablePasses = foundNewProcess ? 0 : stablePasses + 1;
            if (stablePasses >= 2)
            {
                break;
            }

            // The worker is still blocked waiting for its stdin request, so a
            // short settling interval safely closes the venv-launcher race.
            Thread.Sleep(20);
        }
    }

    public void Terminate()
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            return;
        }

        if (!TerminateJobObject(_handle, unchecked((uint)-1)))
        {
            var error = Marshal.GetLastWin32Error();
            if (error != ErrorInvalidParameter)
            {
                throw new Win32Exception(error, "Unable to terminate the local speech Windows Job Object.");
            }
        }
    }

    public async Task WaitForEmptyAsync(CancellationToken cancellationToken = default)
    {
        while (ActiveProcessCount() > 0)
        {
            await Task.Delay(25, cancellationToken).ConfigureAwait(false);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _handle.Dispose();
    }

    private void AssignProcess(int processId)
    {
        Process? process = null;
        try
        {
            process = Process.GetProcessById(processId);
            if (process.HasExited)
            {
                return;
            }

            if (AssignProcessToJobObject(_handle, process.Handle))
            {
                return;
            }

            var error = Marshal.GetLastWin32Error();
            if (process.HasExited)
            {
                return;
            }

            if (IsProcessInJob(process.Handle, _handle, out var alreadyAssigned) && alreadyAssigned)
            {
                return;
            }

            throw new Win32Exception(error, $"Unable to contain local speech process {processId} in its Windows Job Object.");
        }
        catch (ArgumentException)
        {
            // The short-lived venv launcher can exit while its managed Python
            // child is being enumerated. Its child is handled by the snapshot.
        }
        catch (InvalidOperationException)
        {
            // Process exited between GetProcessById and opening its handle.
        }
        finally
        {
            process?.Dispose();
        }
    }

    private uint ActiveProcessCount()
    {
        ThrowIfDisposed();
        var size = Marshal.SizeOf<JobObjectBasicAccountingInformation>();
        var pointer = Marshal.AllocHGlobal(size);
        try
        {
            if (!QueryInformationJobObject(
                    _handle,
                    JobObjectInfoType.BasicAccountingInformation,
                    pointer,
                    (uint)size,
                    out _))
            {
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "Unable to verify that the local speech Windows Job Object is empty.");
            }

            return Marshal.PtrToStructure<JobObjectBasicAccountingInformation>(pointer).ActiveProcesses;
        }
        finally
        {
            Marshal.FreeHGlobal(pointer);
        }
    }

    private static IReadOnlyList<int> DescendantProcessIds(int rootProcessId)
    {
        using var snapshot = CreateToolhelp32Snapshot(Th32csSnapProcess, 0);
        if (snapshot.IsInvalid)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error(), "Unable to inspect the local speech process tree.");
        }

        var childrenByParent = new Dictionary<int, List<int>>();
        var entry = new ProcessEntry32 { Size = (uint)Marshal.SizeOf<ProcessEntry32>() };
        if (Process32First(snapshot, ref entry))
        {
            do
            {
                var parent = unchecked((int)entry.ParentProcessId);
                var process = unchecked((int)entry.ProcessId);
                if (!childrenByParent.TryGetValue(parent, out var children))
                {
                    children = new List<int>();
                    childrenByParent[parent] = children;
                }
                children.Add(process);
                entry.Size = (uint)Marshal.SizeOf<ProcessEntry32>();
            }
            while (Process32Next(snapshot, ref entry));

            var error = Marshal.GetLastWin32Error();
            if (error != 0 && error != ErrorNoMoreFiles)
            {
                throw new Win32Exception(error, "Unable to finish inspecting the local speech process tree.");
            }
        }

        var result = new List<int> { rootProcessId };
        var queue = new Queue<int>();
        queue.Enqueue(rootProcessId);
        while (queue.TryDequeue(out var parent))
        {
            if (!childrenByParent.TryGetValue(parent, out var children))
            {
                continue;
            }

            foreach (var child in children)
            {
                if (result.Contains(child))
                {
                    continue;
                }
                result.Add(child);
                queue.Enqueue(child);
            }
        }
        return result;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    private enum JobObjectInfoType
    {
        BasicAccountingInformation = 1,
        ExtendedLimitInformation = 9
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectBasicAccountingInformation
    {
        public long TotalUserTime;
        public long TotalKernelTime;
        public long ThisPeriodTotalUserTime;
        public long ThisPeriodTotalKernelTime;
        public uint TotalPageFaultCount;
        public uint TotalProcesses;
        public uint ActiveProcesses;
        public uint TotalTerminatedProcesses;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectBasicLimitInformation
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectExtendedLimitInformation
    {
        public JobObjectBasicLimitInformation BasicLimitInformation;
        public IoCounters IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct ProcessEntry32
    {
        public uint Size;
        public uint Usage;
        public uint ProcessId;
        public UIntPtr DefaultHeapId;
        public uint ModuleId;
        public uint Threads;
        public uint ParentProcessId;
        public int BasePriority;
        public uint Flags;

        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
        public string ExecutableFile;
    }

    private sealed class SafeJobHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        private SafeJobHandle()
            : base(ownsHandle: true)
        {
        }

        protected override bool ReleaseHandle() => CloseHandle(handle);
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern SafeJobHandle CreateJobObject(IntPtr jobAttributes, string? name);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(
        SafeJobHandle job,
        JobObjectInfoType informationClass,
        IntPtr information,
        uint informationLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool QueryInformationJobObject(
        SafeJobHandle job,
        JobObjectInfoType informationClass,
        IntPtr information,
        uint informationLength,
        out uint returnLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(SafeJobHandle job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool IsProcessInJob(IntPtr process, SafeJobHandle job, out bool result);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateJobObject(SafeJobHandle job, uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern SafeFileHandle CreateToolhelp32Snapshot(uint flags, uint processId);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool Process32First(SafeFileHandle snapshot, ref ProcessEntry32 entry);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool Process32Next(SafeFileHandle snapshot, ref ProcessEntry32 entry);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);
}
