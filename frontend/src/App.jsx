import React, { useState, useCallback, useEffect, createContext, useContext, useRef } from 'react';
import {
    FileText, Settings, Play, BarChart2, Cpu, FlaskConical, FolderOpen,
    BrainCircuit, X, Loader2, Bot, KeyRound, Server, RefreshCw, CheckCircle,
    AlertTriangle, PlusCircle, FolderUp, File, Home // Added FolderUp, File, Home
} from 'lucide-react';
import { Toaster, toast } from 'sonner'; // Using sonner for toasts

// --- Shadcn/ui Component Mocks (Using basic HTML elements + Tailwind) ---
// These are simplified versions. For full features, use the actual libraries.

const Button = React.forwardRef(({ children, variant = 'default', size = 'default', className = '', disabled, ...props }, ref) => (
    <button
        ref={ref}
        className={`inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ${
            variant === 'destructive' ? 'bg-red-500 text-destructive-foreground hover:bg-red-600/90' :
            variant === 'outline' ? 'border border-input bg-background hover:bg-accent hover:text-accent-foreground' :
            variant === 'secondary' ? 'bg-secondary text-secondary-foreground hover:bg-secondary/80' :
            variant === 'ghost' ? 'hover:bg-accent hover:text-accent-foreground' :
            variant === 'link' ? 'text-primary underline-offset-4 hover:underline' :
            'bg-blue-600 text-primary-foreground hover:bg-blue-700/90' // Default variant
        } ${
            size === 'sm' ? 'h-9 px-3' :
            size === 'lg' ? 'h-11 px-8' :
            size === 'icon' ? 'h-10 w-10' :
            'h-10 px-4 py-2' // Default size
        } ${className}`}
        disabled={disabled}
        {...props}
    >
        {children}
    </button>
));
Button.displayName = 'Button';


const Input = React.forwardRef(({ className = '', type, disabled, ...props }, ref) => (
    <input
        type={type}
        ref={ref}
        className={`flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
        disabled={disabled}
        {...props}
    />
));
Input.displayName = 'Input';

const Label = React.forwardRef(({ className = '', ...props }, ref) => (
    <label
        ref={ref}
        className={`block text-sm font-medium text-gray-700 leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 ${className}`}
        {...props}
    />
));
Label.displayName = 'Label';

const Textarea = React.forwardRef(({ className = '', disabled, ...props }, ref) => (
    <textarea
        ref={ref}
        className={`flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
        disabled={disabled}
        {...props}
    />
));
Textarea.displayName = 'Textarea';

const Card = React.forwardRef(({ className = '', children, ...props }, ref) => (
    <div ref={ref} className={`rounded-lg border bg-card text-card-foreground shadow-sm ${className}`} {...props}>
        {children}
    </div>
));
Card.displayName = 'Card';

const CardHeader = React.forwardRef(({ className = '', children, ...props }, ref) => (
    <div ref={ref} className={`flex flex-col space-y-1.5 p-4 md:p-6 ${className}`} {...props}>
        {children}
    </div>
));
CardHeader.displayName = 'CardHeader';

const CardTitle = React.forwardRef(({ className = '', children, ...props }, ref) => (
    <h3 ref={ref} className={`text-lg font-semibold leading-none tracking-tight ${className}`} {...props}>
        {children}
    </h3>
));
CardTitle.displayName = 'CardTitle';

const CardDescription = React.forwardRef(({ className = '', children, ...props }, ref) => (
    <p ref={ref} className={`text-sm text-muted-foreground ${className}`} {...props}>
        {children}
    </p>
));
CardDescription.displayName = 'CardDescription';

const CardContent = React.forwardRef(({ className = '', children, ...props }, ref) => (
    <div ref={ref} className={`p-4 md:p-6 pt-0 ${className}`} {...props}>
        {children}
    </div>
));
CardContent.displayName = 'CardContent';

const CardFooter = React.forwardRef(({ className = '', children, ...props }, ref) => (
    <div ref={ref} className={`flex items-center p-4 md:p-6 pt-0 ${className}`} {...props}>
        {children}
    </div>
));
CardFooter.displayName = 'CardFooter';

// Basic Select Mock (Not fully functional like Shadcn's)
const Select = React.forwardRef(({ children, className = '', disabled, value, onChange, ...props }, ref) => (
    <select
        ref={ref}
        value={value}
        onChange={onChange}
        className={`flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1 ${className}`}
        disabled={disabled}
        {...props}
    >
        {children}
    </select>
));
Select.displayName = 'Select';

const SelectItem = ({ children, value, disabled, ...props }) => (
    <option value={value} disabled={disabled} {...props}>{children}</option>
);

// Basic MultiSelect Placeholder Mock
const MultiSelect = ({ options = [], selected = [], onChange, placeholder = "Select...", className = '', disabled }) => {
    // This is just a placeholder visually. A real multi-select is complex.
    // We'll simulate selection by clicking options.
    const handleToggle = (optionValue) => {
        if (disabled) return;
        const newSelected = selected.includes(optionValue)
            ? selected.filter(v => v !== optionValue)
            : [...selected, optionValue];
        onChange(newSelected);
    };

    return (
        <div className={`p-2 border rounded-md bg-gray-50 min-h-[40px] ${className} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}>
            <span className="text-sm text-gray-600">
                {selected.length > 0 ? selected.join(', ') : placeholder}
            </span>
            <div className="mt-2 space-y-1 max-h-32 overflow-y-auto text-xs">
                {options.map(option => (
                    <button
                        key={option}
                        type="button"
                        onClick={() => handleToggle(option)}
                        className={`block w-full text-left p-1 rounded ${selected.includes(option) ? 'bg-blue-100' : 'hover:bg-gray-100'} ${disabled ? 'pointer-events-none' : ''}`}
                        disabled={disabled}
                    >
                        {option}
                    </button>
                ))}
            </div>
            <p className="text-xs text-gray-400 mt-1">(Mock multi-select: click options above)</p>
        </div>
    );
};

const ScrollArea = React.forwardRef(({ children, className = '' }, ref) => (
    // Simple overflow container
    <div ref={ref} className={`overflow-auto ${className}`}>
        {children}
    </div>
));
ScrollArea.displayName = 'ScrollArea';


// --- App Context ---
const AppContext = createContext();

export const AppProvider = ({ children }) => {
    const [currentProject, setCurrentProject] = useState(null);
    const [projects, setProjects] = useState([]);
    const [workingDirectory, setWorkingDirectory] = useState('.');
    const [llmConfig, setLlmConfig] = useState({
        provider: 'openai',
        apiKey: '', // Store API key state here temporarily for input, but don't persist/expose unnecessarily
        apiModel: 'gpt-4o',
        ollamaModel: 'llama3', // Default Ollama model
        ollamaEndpoint: 'http://localhost:11434', // Default Ollama endpoint
    });
    const [availableOllamaModels, setAvailableOllamaModels] = useState([]);
    const [configStatus, setConfigStatus] = useState({ llm: 'pending', cwd: 'pending' }); // 'ok', 'error', 'pending'

    const updateWorkingDirectory = useCallback((newDir) => setWorkingDirectory(newDir), []);
    const updateLlmConfig = useCallback((newConfig) => setLlmConfig(prev => ({ ...prev, ...newConfig })), []);
    const updateConfigStatus = useCallback((type, status) => setConfigStatus(prev => ({ ...prev, [type]: status })), []);

    // Fetch project list from backend
    const loadProjects = useCallback(async (addLog) => {
        addLog("Fetching project list...");
        try {
            const data = await callBackendApi('/projects');
            setProjects(data.projects || []);
            addLog(`Found ${data.projects?.length || 0} projects.`);
        } catch (error) {
            addLog(`Error fetching projects: ${error.message}`, 'error');
            setProjects([]);
            toast.error(`Failed to load projects: ${error.message}`);
        }
    }, []);

    // Select a project (inform backend)
    const selectProject = useCallback(async (projectName, addLog) => {
        if (!projectName) {
            setCurrentProject(null);
            addLog("No project selected.");
            // Optionally, inform backend that no project is active
            // await callBackendApi('/unload_project', 'POST');
            return;
        }
        addLog(`Selecting project: ${projectName}...`);
        try {
            // Use the /set_working_directory endpoint to also load the project context
            const response = await callBackendApi('/set_working_directory', 'POST', { path: projectName });
             if (response.success) {
                 setCurrentProject(projectName);
                 updateWorkingDirectory(response.path); // Update CWD based on backend response
                 addLog(`Project '${projectName}' selected. Working directory set to: ${response.path}`, 'success');
                 toast.success(`Project '${projectName}' selected.`);
                 updateConfigStatus('cwd', 'ok'); // Assume CWD is ok if project selection worked
             } else {
                 throw new Error(response.message || "Failed to select project via setting working directory");
             }
        } catch (error) {
            addLog(`Error selecting project '${projectName}': ${error.message}`, 'error');
            toast.error(`Failed to select project: ${error.message}`);
            setCurrentProject(null); // Reset project selection on error
            updateConfigStatus('cwd', 'error');
        }
    }, [updateWorkingDirectory, updateConfigStatus]); // Added dependencies

    // Create a new project (inform backend)
    const createProject = useCallback(async (projectName, addLog) => {
        addLog(`Creating project: ${projectName}...`);
        try {
            // Backend should create the directory and necessary structure
            const response = await callBackendApi('/create_project', 'POST', { name: projectName });
            if (response.success) {
                addLog(`Project '${projectName}' created successfully. Path: ${response.path}`, 'success');
                toast.success(`Project '${projectName}' created.`);
                await loadProjects(addLog); // Refresh project list
                await selectProject(response.path, addLog); // Select the newly created project (use path from response)
                return response.path; // Return the path for potential immediate use
            } else {
                throw new Error(response.message || "Failed to create project");
            }
        } catch (error) {
            addLog(`Error creating project '${projectName}': ${error.message}`, 'error');
            toast.error(`Project creation failed: ${error.message}`);
            throw error; // Re-throw error so UI knows it failed
        }
    }, [loadProjects, selectProject]); // Added dependencies

    const value = {
        currentProject,
        selectProject,
        projects,
        loadProjects,
        createProject,
        workingDirectory,
        updateWorkingDirectory,
        llmConfig,
        updateLlmConfig,
        availableOllamaModels,
        setAvailableOllamaModels,
        configStatus,
        updateConfigStatus,
    };

    return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useAppContext = () => useContext(AppContext);


// --- API Helper (Using Fetch) ---
const API_BASE_URL = 'http://localhost:8000/api'; // Your FastAPI backend URL

async function callBackendApi(endpoint, method = 'GET', body = null) {
    const url = `${API_BASE_URL}${endpoint}`;
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
            // Add any other headers like Authorization if needed
        },
    };

    if (body && method !== 'GET') {
        options.body = JSON.stringify(body);
    }

    // Handle GET requests with query parameters if body is provided
    let fetchUrl = url;
    if (body && method === 'GET') {
        const params = new URLSearchParams(body);
        fetchUrl = `${url}?${params.toString()}`;
    }

    console.log(`Calling backend: ${method} ${fetchUrl}`, body); // Log the actual URL being fetched

    try {
        const response = await fetch(fetchUrl, options);

        if (!response.ok) {
            // Try to parse error message from backend response body
            let errorBody;
            try {
                errorBody = await response.json();
            } catch (parseError) {
                // If response body is not JSON or empty
                throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
            }
            // Use detailed error from backend if available
            const detail = errorBody?.detail || `HTTP error! status: ${response.status} ${response.statusText}`;
            throw new Error(detail);
        }

        // Handle cases where response might be empty (e.g., 204 No Content)
        if (response.status === 204) {
            return null; // Or return an empty object/success indicator
        }

        const data = await response.json();
        console.log("Backend response:", data);
        return data;

    } catch (error) {
        console.error("API call failed:", error);
        // Don't show toast here, let the calling component handle UI feedback
        // toast.error(`API Error: ${error.message}`);
        throw error; // Re-throw the error so the caller knows it failed
    }
}


// --- Components ---

// Project Panel Component
const ProjectPanel = ({ addLog }) => {
    const { currentProject, selectProject, projects, loadProjects, createProject } = useAppContext();
    const [newProjectName, setNewProjectName] = useState('');
    const [isCreating, setIsCreating] = useState(false);
    const [createLoading, setCreateLoading] = useState(false);

    // Load projects on initial mount
    useEffect(() => {
        loadProjects(addLog);
    }, [loadProjects, addLog]);

    const handleCreateToggle = () => {
        setIsCreating(!isCreating);
        setNewProjectName(''); // Clear input when toggling
    };

    const handleCreateSubmit = async () => {
        const trimmedName = newProjectName.trim();
        if (trimmedName) {
            setCreateLoading(true);
            try {
                await createProject(trimmedName, addLog);
                setIsCreating(false); // Close create form on success
                setNewProjectName(''); // Clear input
            } catch (error) {
                // Error is logged and toasted in createProject/callBackendApi
                // Keep the form open for correction
            } finally {
                setCreateLoading(false);
            }
        } else {
            toast.warning("Please enter a name for the new project.");
        }
    };

    return (
        <div className="p-2 border-b space-y-2 bg-gray-50"> {/* Added subtle bg */}
            <div>
                <Label htmlFor="projectSelect" className="text-xs font-medium text-gray-500 mb-1">Project</Label>
                <div className="flex gap-1">
                    <Select
                        id="projectSelect"
                        value={currentProject || ""}
                        onChange={(e) => selectProject(e.target.value, addLog)}
                        className="flex-grow h-8 text-sm px-2 py-1 rounded-md" // Ensure consistent styling
                        disabled={isCreating || !projects.length} // Disable if creating or no projects
                    >
                        <SelectItem value="" disabled>
                            {projects.length === 0 ? "No projects found" : "Select Project"}
                        </SelectItem>
                        {projects.map(p => <SelectItem key={p} value={p}>{p}</SelectItem>)}
                    </Select>
                    <Button
                        onClick={() => loadProjects(addLog)}
                        variant="outline"
                        size="icon"
                        className="h-8 w-8 flex-shrink-0"
                        disabled={isCreating}
                        aria-label="Refresh project list"
                    >
                        <RefreshCw className="h-4 w-4" />
                    </Button>
                </div>
            </div>

            {/* Create Project Section */}
            <div>
                {isCreating ? (
                    <div className="space-y-1 p-2 border rounded bg-gray-100">
                        <Label htmlFor="newProjectName" className="text-xs font-medium text-gray-600">New Project Name</Label>
                        <div className="flex gap-1">
                            <Input
                                id="newProjectName"
                                type="text"
                                value={newProjectName}
                                onChange={e => setNewProjectName(e.target.value)}
                                placeholder="Enter name..."
                                className="flex-grow h-8 text-sm px-2 py-1" // Consistent styling
                                disabled={createLoading}
                                autoFocus
                            />
                            <Button
                                onClick={handleCreateSubmit}
                                size="sm"
                                className="h-8 text-xs flex-shrink-0 px-2" // Adjusted padding
                                disabled={createLoading}
                            >
                                {createLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Create"}
                            </Button>
                        </div>
                        <Button
                            onClick={handleCreateToggle}
                            variant="link"
                            size="sm"
                            className="text-xs p-0 h-auto text-muted-foreground" // Subtle cancel
                            disabled={createLoading}
                        >
                            Cancel
                        </Button>
                    </div>
                ) : (
                    <Button
                        onClick={handleCreateToggle}
                        variant="outline"
                        size="sm"
                        className="w-full h-8 text-xs justify-center" // Centered text
                    >
                        <PlusCircle className="h-4 w-4 mr-1" /> Create New Project
                    </Button>
                )}
            </div>
        </div>
    );
};


// Sidebar Navigation Component
const Sidebar = ({ currentView, setView, addLog }) => {
    const { configStatus } = useAppContext(); // Get status from context

    // Navigation items configuration
    const navItems = [
        { id: 'files', label: 'File Explorer', icon: FolderOpen },
        { id: 'preprocess', label: 'Preprocess Data', icon: FlaskConical },
        { id: 'train', label: 'Train Model', icon: BrainCircuit },
        { id: 'evaluate', label: 'Evaluate Model', icon: BarChart2 },
        { id: 'plot', label: 'Plot Results', icon: BarChart2 }, // Changed icon for consistency
        { id: 'custom', label: 'Custom Instruction', icon: Cpu },
        { id: 'autopilot', label: 'Auto-Pilot', icon: Play },
    ];

    // Helper to get status icon based on configStatus
    const getStatusIcon = (status) => {
        switch (status) {
            case 'ok': return <CheckCircle className="h-4 w-4 text-green-500" aria-label="OK"/>;
            case 'error': return <AlertTriangle className="h-4 w-4 text-red-500" aria-label="Error"/>;
            default: return <Loader2 className="h-4 w-4 text-yellow-500 animate-spin" aria-label="Pending"/>; // pending or unknown
        }
    };

    return (
        <aside className="w-64 bg-gray-50 border-r border-gray-200 p-0 flex flex-col">
            {/* Header */}
            <div className="flex items-center gap-2 p-4 border-b">
                <Bot size={28} className="text-blue-600" />
                <h1 className="text-xl font-semibold text-gray-800">ML Copilot</h1>
            </div>

            {/* Project Panel */}
            <ProjectPanel addLog={addLog} />

            {/* Main Navigation */}
            <nav className="flex-grow space-y-1 p-4 overflow-y-auto">
                {navItems.map(item => (
                    <Button
                        key={item.id}
                        variant={currentView === item.id ? 'secondary' : 'ghost'}
                        className={`w-full justify-start ${
                            currentView === item.id
                                ? 'bg-blue-100 text-blue-700' // Active state
                                : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900' // Inactive state
                        }`}
                        onClick={() => setView(item.id)}
                    >
                        <item.icon className="mr-2 h-4 w-4 flex-shrink-0" /> {/* Ensure icon doesn't shrink */}
                        {item.label}
                    </Button>
                ))}
            </nav>

            {/* Footer Navigation (Settings) */}
            <div className="p-4 border-t mt-auto">
                <Button
                    variant={currentView === 'settings' ? 'secondary' : 'ghost'}
                    className={`w-full justify-start ${
                        currentView === 'settings'
                            ? 'bg-blue-100 text-blue-700'
                            : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                    }`}
                    onClick={() => setView('settings')}
                >
                    <Settings className="mr-2 h-4 w-4 flex-shrink-0" />
                    Settings
                    {/* Status Icons */}
                    <div className="ml-auto flex items-center gap-1" title={`CWD: ${configStatus.cwd}, LLM: ${configStatus.llm}`}>
                        {getStatusIcon(configStatus.cwd)}
                        {getStatusIcon(configStatus.llm)}
                    </div>
                </Button>
            </div>
        </aside>
    );
};

// Main Content Area Router Component
const MainContent = ({ currentView, addLog, clearLogs }) => {
    const { currentProject } = useAppContext();

    // Views that require a project to be selected
    const projectRequiredViews = ['files', 'preprocess', 'train', 'evaluate', 'plot', 'custom', 'autopilot'];

    // Show message if a project is required but not selected
    if (!currentProject && projectRequiredViews.includes(currentView)) {
        return (
            <Card className="m-4">
                <CardHeader><CardTitle>No Project Selected</CardTitle></CardHeader>
                <CardContent>
                    <p>Please select or create a project using the panel on the left to access this view.</p>
                </CardContent>
            </Card>
        );
    }

    // Render the appropriate view based on currentView
    switch (currentView) {
        case 'files': return <FilesView addLog={addLog} />;
        case 'preprocess': return <PreprocessView addLog={addLog} />;
        case 'train': return <TrainView addLog={addLog} />;
        case 'evaluate': return <EvaluateView addLog={addLog} />;
        case 'plot': return <PlotView addLog={addLog} />;
        case 'custom': return <CustomInstructionView addLog={addLog} />;
        case 'autopilot': return <AutoPilotView addLog={addLog} />;
        case 'settings': return <SettingsView addLog={addLog} />;
        case 'welcome': // Default/Welcome view
        default:
            return (
                <Card className="m-4">
                    <CardHeader><CardTitle>Welcome to ML Copilot Agent GUI</CardTitle></CardHeader>
                    <CardContent className="space-y-4">
                        <p>Select a project, configure settings, or choose an action from the sidebar to get started.</p>
                        <p className="text-sm text-muted-foreground">Use the <Settings className="inline h-4 w-4 mx-1"/>Settings panel to configure your working directory and Language Model (LLM).</p>
                        <Button onClick={clearLogs} variant="outline" size="sm">
                            <X className="mr-1 h-3 w-3"/> Clear Logs
                        </Button>
                    </CardContent>
                </Card>
            );
    }
};

// --- View Components ---

// FilesView Component
const FilesView = ({ addLog }) => {
    const { workingDirectory } = useAppContext(); // Base directory from context
    const [files, setFiles] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [currentRelativePath, setCurrentRelativePath] = useState('.'); // Path relative to workingDirectory

    // Function to fetch files for a given relative path
    const fetchFiles = useCallback(async (relativePath) => {
        setLoading(true);
        setError(null);
        addLog(`Fetching file list for path: ${relativePath} (relative to ${workingDirectory})`);
        try {
            // Backend expects path relative to its CWD, which should match our workingDirectory
            const data = await callBackendApi('/files', 'GET', { path: relativePath });
            // Sort files: directories first, then files, alphabetically
            const sortedFiles = (data.files || []).sort((a, b) => {
                const isADir = a.endsWith('/');
                const isBDir = b.endsWith('/');
                if (isADir && !isBDir) return -1;
                if (!isADir && isBDir) return 1;
                return a.localeCompare(b);
            });
            setFiles(sortedFiles);
            setCurrentRelativePath(data.current_directory || relativePath); // Use path confirmed by backend
            addLog(`Found ${sortedFiles.length} items in ${data.current_directory || relativePath}.`);
        } catch (err) {
            setError(err.message);
            addLog(`Error fetching files for ${relativePath}: ${err.message}`, 'error');
            toast.error(`Failed to load files: ${err.message}`);
            setFiles([]); // Clear files on error
        } finally {
            setLoading(false);
        }
    }, [addLog, workingDirectory]); // Include dependencies

    // Fetch files for the root directory on mount and when workingDirectory changes
    useEffect(() => {
        setCurrentRelativePath('.'); // Reset to root when WD changes
        fetchFiles('.');
    }, [fetchFiles, workingDirectory]); // Add workingDirectory dependency

    // Handle clicking on a file or directory item
    const handleItemClick = (item) => {
        if (item.endsWith('/')) {
            // It's a directory, navigate into it
            const dirName = item.slice(0, -1);
            // Construct the new relative path
            const newPath = currentRelativePath === '.' ? dirName : `${currentRelativePath}/${dirName}`;
            fetchFiles(newPath);
        } else {
            // It's a file, log selection (or implement file view/edit later)
            const fullPath = currentRelativePath === '.' ? item : `${currentRelativePath}/${item}`;
            addLog(`Selected file: ${fullPath}`);
            toast.info(`Selected file: ${fullPath}`);
        }
    };

    // Handle clicking the "Go Up" button
    const handleGoUp = () => {
        if (currentRelativePath === '.') return; // Already at root

        // Find the last '/' to go up one level
        const parts = currentRelativePath.split('/');
        parts.pop(); // Remove the last part
        const newPath = parts.length === 0 ? '.' : parts.join('/');
        fetchFiles(newPath);
    };

    // Handle clicking the "Go Home" button
    const handleGoHome = () => {
        if (currentRelativePath !== '.') {
            fetchFiles('.');
        }
    };


    return (
        <Card className="m-4">
            <CardHeader>
                <CardTitle>File Explorer</CardTitle>
                <CardDescription>
                    Base Directory: <code className="bg-gray-100 px-1 rounded text-xs">{workingDirectory}</code>
                    <br />
                    Current Path: <code className="bg-gray-100 px-1 rounded text-xs">{currentRelativePath}</code>
                </CardDescription>
            </CardHeader>
            <CardContent>
                {/* Action Buttons */}
                <div className="flex gap-2 mb-3 border-b pb-3">
                    <Button onClick={handleGoHome} disabled={loading || currentRelativePath === '.'} variant="outline" size="sm" aria-label="Go to base directory">
                        <Home className="mr-1 h-4 w-4" /> Home
                    </Button>
                    <Button onClick={handleGoUp} disabled={loading || currentRelativePath === '.'} variant="outline" size="sm" aria-label="Go up one directory">
                        <FolderUp className="mr-1 h-4 w-4" /> Up
                    </Button>
                    <Button onClick={() => fetchFiles(currentRelativePath)} disabled={loading} variant="outline" size="sm" className="ml-auto" aria-label="Refresh current directory">
                        <RefreshCw className={`mr-1 h-4 w-4 ${loading ? 'animate-spin' : ''}`} /> Refresh
                    </Button>
                </div>

                {/* Loading/Error State */}
                {loading && <div className="flex justify-center my-4"><Loader2 className="h-6 w-6 animate-spin text-blue-500" /></div>}
                {error && <p className="text-red-600 text-sm p-2 bg-red-50 rounded border border-red-200">Error: {error}</p>}

                {/* File List */}
                {!loading && !error && (
                    <ScrollArea className="h-72 border rounded-md p-2 bg-gray-50"> {/* Increased height */}
                        {files.length > 0 ? (
                            <ul>
                                {files.map((item, index) => (
                                    <li key={index} className="text-sm py-1 px-1 group">
                                        <button
                                            onClick={() => handleItemClick(item)}
                                            className={`flex items-center text-left w-full rounded group-hover:bg-blue-50 p-1 ${
                                                item.endsWith('/') ? 'text-yellow-700 font-medium' : 'text-gray-800'
                                            }`}
                                        >
                                            {item.endsWith('/')
                                                ? <FolderOpen size={16} className="mr-2 flex-shrink-0 text-yellow-600"/>
                                                : <File size={16} className="mr-2 text-gray-500 flex-shrink-0"/> // Generic file icon
                                            }
                                            <span className="truncate">{item}</span> {/* Prevent long names breaking layout */}
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p className="text-gray-500 text-sm italic p-4 text-center">Directory is empty.</p>
                        )}
                    </ScrollArea>
                )}
            </CardContent>
        </Card>
    );
};


// PreprocessView Component
const PreprocessView = ({ addLog }) => {
    const { workingDirectory } = useAppContext();
    const [datasetPath, setDatasetPath] = useState(''); // Relative path selected via browser
    const [availableColumns, setAvailableColumns] = useState([]);
    const [targetColumn, setTargetColumn] = useState('');
    const [columnsToDrop, setColumnsToDrop] = useState([]);
    const [savePath, setSavePath] = useState('data/preprocessed_data.csv'); // Default relative save path
    const [instructions, setInstructions] = useState('');
    const [loading, setLoading] = useState(false); // For main preprocess action
    const [columnsLoading, setColumnsLoading] = useState(false); // For fetching columns

    // State for the mini file browser
    const [browsePath, setBrowsePath] = useState('.'); // Relative path for the browser
    const [browseFiles, setBrowseFiles] = useState([]);
    const [browseLoading, setBrowseLoading] = useState(false);

    // Fetch files for the mini browser
    const fetchFilesForBrowse = useCallback(async (relativePath) => {
        setBrowseLoading(true);
        try {
            const data = await callBackendApi('/files', 'GET', { path: relativePath });
             const sortedFiles = (data.files || []).sort((a, b) => {
                 const isADir = a.endsWith('/');
                 const isBDir = b.endsWith('/');
                 if (isADir && !isBDir) return -1;
                 if (!isADir && isBDir) return 1;
                 return a.localeCompare(b);
             });
            setBrowseFiles(sortedFiles);
            setBrowsePath(data.current_directory || relativePath);
        } catch (error) {
            toast.error(`Failed to browse files: ${error.message}`);
            setBrowseFiles([]); // Clear on error
        } finally {
            setBrowseLoading(false);
        }
    }, []); // No dependencies needed here as it uses its own path state

    // Fetch columns when datasetPath changes
    useEffect(() => {
        if (!datasetPath) {
            setAvailableColumns([]);
            setTargetColumn('');
            setColumnsToDrop([]); // Reset dependent fields
            return;
        }

        const fetchColumns = async () => {
            setColumnsLoading(true);
            setAvailableColumns([]);
            setTargetColumn('');
            setColumnsToDrop([]);
            addLog(`Fetching columns for: ${datasetPath}`);
            try {
                // Backend expects path relative to its CWD
                const data = await callBackendApi('/get_columns', 'GET', { file_path: datasetPath });
                setAvailableColumns(data.columns || []);
                addLog(`Found columns: ${data.columns?.join(', ') || 'None'}`);
                if (!data.columns || data.columns.length === 0) {
                    toast.warning(`No columns found in ${datasetPath}. Is it a valid CSV/data file?`);
                }
            } catch (error) {
                addLog(`Error fetching columns for ${datasetPath}: ${error.message}`, 'error');
                toast.error(`Could not get columns: ${error.message}`);
            } finally {
                setColumnsLoading(false);
            }
        };

        fetchColumns();
    }, [datasetPath, addLog]); // Rerun when datasetPath changes

    // Initial fetch for the browser
    useEffect(() => {
        fetchFilesForBrowse('.');
    }, [fetchFilesForBrowse]);

    // Handle selecting a file in the mini browser
    const handleFileSelect = (item) => {
        if (item.endsWith('/')) {
            // Navigate into directory
            const dirName = item.slice(0, -1);
            const newPath = browsePath === '.' ? dirName : `${browsePath}/${dirName}`;
            fetchFilesForBrowse(newPath);
        } else {
            // Select the file
            const selectedFilePath = browsePath === '.' ? item : `${browsePath}/${item}`;
            // Only allow CSV or common data file extensions
             if (selectedFilePath.match(/\.(csv|data|tsv|txt)$/i)) {
                setDatasetPath(selectedFilePath);
                addLog(`Selected dataset for preprocessing: ${selectedFilePath}`);
             } else {
                 toast.warning("Please select a valid data file (e.g., .csv, .data, .tsv).");
             }
        }
    };

    // Handle going up in the mini browser
    const handleBrowseUp = () => {
        if (browsePath === '.') return;
        const parts = browsePath.split('/');
        parts.pop();
        fetchFilesForBrowse(parts.length === 0 ? '.' : parts.join('/'));
    };

    // Handle form submission
    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!datasetPath || !targetColumn) {
            toast.warning("Dataset Path and Target Column are required.");
            return;
        }
        setLoading(true);
        const effectiveSavePath = savePath || 'data/preprocessed_data.csv'; // Ensure default if empty
        addLog(`Starting preprocessing: Dataset='${datasetPath}', Target='${targetColumn}', Save='${effectiveSavePath}'`);
        try {
            const result = await callBackendApi('/preprocess', 'POST', {
                dataset_path: datasetPath,
                target_column: targetColumn,
                save_path: effectiveSavePath,
                columns_to_drop: columnsToDrop,
                additional_instructions: instructions,
            });
            addLog(`Preprocessing task submitted. Agent response: ${result.message}`, 'success');
            toast.success("Preprocessing task submitted successfully.");
            // Optionally clear form or update state based on success
        } catch (err) {
            addLog(`Preprocessing failed: ${err.message}`, 'error');
            toast.error(`Preprocessing failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Card className="m-4">
            <CardHeader>
                <CardTitle>Preprocess Data</CardTitle>
                <CardDescription>
                    Select dataset, target, columns to drop, and specify preprocessing steps.
                    Base directory: <code className="text-xs">{workingDirectory}</code>
                </CardDescription>
            </CardHeader>
            <form onSubmit={handleSubmit}>
                <CardContent className="space-y-4">
                    {/* Dataset File Browser */}
                    <div>
                        <Label className="mb-1">Select Dataset File *</Label>
                        <div className="p-2 border rounded-md bg-gray-50 space-y-1">
                            <div className="flex gap-1 text-xs items-center mb-1">
                                <span className="font-medium">Current:</span>
                                <code className="bg-white px-1 rounded text-xs flex-grow truncate">{browsePath}</code>
                                <Button onClick={handleBrowseUp} disabled={browseLoading || browsePath === '.'} variant="ghost" size="sm" className="ml-auto h-6 px-1 text-xs">.. Up</Button>
                                <Button onClick={() => fetchFilesForBrowse(browsePath)} variant="ghost" size="icon" className="h-6 w-6" disabled={browseLoading} aria-label="Refresh browser">
                                    <RefreshCw className={`h-3 w-3 ${browseLoading ? 'animate-spin' : ''}`} />
                                </Button>
                            </div>
                            <ScrollArea className="h-32 border rounded bg-white p-1">
                                {browseLoading ? <div className="flex justify-center p-2"><Loader2 className="h-4 w-4 animate-spin"/></div> : (
                                    browseFiles.length > 0 ? (
                                        <ul>
                                            {browseFiles.map((item, i) => (
                                                <li key={i}>
                                                    <button
                                                        type="button"
                                                        onClick={() => handleFileSelect(item)}
                                                        className={`flex items-center text-xs w-full text-left p-0.5 rounded hover:bg-blue-50 group ${
                                                            item.endsWith('/') ? 'text-yellow-700' : 'text-gray-800'
                                                        }`}
                                                    >
                                                        {item.endsWith('/')
                                                            ? <FolderOpen size={14} className="mr-1 shrink-0 text-yellow-600"/>
                                                            : <File size={14} className="mr-1 shrink-0 text-gray-500"/>}
                                                        <span className="truncate">{item}</span>
                                                    </button>
                                                </li>
                                            ))}
                                        </ul>
                                    ) : <p className="text-xs text-gray-400 p-2 text-center italic">Empty or error.</p>
                                )}
                            </ScrollArea>
                            <p className="text-xs text-gray-600 mt-1">
                                Selected: <code className="bg-white px-1 rounded font-medium">{datasetPath || 'None'}</code>
                            </p>
                        </div>
                    </div>

                    {/* Target Column */}
                    <div>
                        <Label htmlFor="targetColumn">Target Column *</Label>
                        <Select
                            id="targetColumn"
                            value={targetColumn}
                            onChange={(e) => setTargetColumn(e.target.value)}
                            disabled={columnsLoading || availableColumns.length === 0}
                            required
                        >
                            <SelectItem value="" disabled>
                                {columnsLoading ? "Loading columns..." :
                                 !datasetPath ? "Select dataset first" :
                                 availableColumns.length === 0 ? "No columns found" :
                                 "Select target column"}
                            </SelectItem>
                            {availableColumns.map(col => <SelectItem key={col} value={col}>{col}</SelectItem>)}
                        </Select>
                    </div>

                    {/* Columns to Drop */}
                    <div>
                        <Label htmlFor="dropColumns">Columns to Drop (Optional)</Label>
                        <MultiSelect
                            options={availableColumns.filter(c => c !== targetColumn)} // Exclude target from drop options
                            selected={columnsToDrop}
                            onChange={setColumnsToDrop} // Pass the setter function directly
                            placeholder="Select columns to remove"
                            disabled={columnsLoading || availableColumns.length === 0}
                            className="w-full" // Ensure it takes full width
                        />
                    </div>

                    {/* Save Path */}
                    <div>
                        <Label htmlFor="savePath">Save Path for Preprocessed Data (relative)</Label>
                        <Input
                            id="savePath"
                            value={savePath}
                            onChange={(e) => setSavePath(e.target.value)}
                            placeholder="Default: data/preprocessed_data.csv"
                        />
                    </div>

                    {/* Additional Instructions */}
                    <div>
                        <Label htmlFor="instructions">Additional Instructions (Optional)</Label>
                        <Textarea
                            id="instructions"
                            value={instructions}
                            onChange={(e) => setInstructions(e.target.value)}
                            placeholder="e.g., use standard scaler, handle missing values with median, perform PCA with 5 components"
                            rows={3}
                        />
                    </div>
                </CardContent>
                <CardFooter>
                    <Button type="submit" disabled={loading || !datasetPath || !targetColumn}>
                        {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                        Start Preprocessing
                    </Button>
                </CardFooter>
            </form>
        </Card>
    );
};


// TrainView Component
const TrainView = ({ addLog }) => {
    const { workingDirectory, currentProject } = useAppContext();
    const [availableDataFiles, setAvailableDataFiles] = useState([]); // Files in data/ dir
    const [dataPathSelection, setDataPathSelection] = useState(''); // Dropdown value ('manual' or file path)
    const [manualDataPath, setManualDataPath] = useState(''); // Manually entered path
    const [targetColumn, setTargetColumn] = useState(''); // Target column name
    const [taskType, setTaskType] = useState('classification'); // 'classification' or 'regression'
    const [modelSavePath, setModelSavePath] = useState(''); // Default set in useEffect
    const [instructions, setInstructions] = useState('');
    const [loading, setLoading] = useState(false); // For main train action
    const [dataFilesLoading, setDataFilesLoading] = useState(false); // For fetching data files

    // Function to fetch available data files (e.g., CSVs in data/)
    const fetchDataFiles = useCallback(async () => {
        setDataFilesLoading(true);
        addLog("Fetching data files from 'data/' directory...");
        setAvailableDataFiles([]); // Clear previous list
        try {
            const data = await callBackendApi('/files', 'GET', { path: 'data' }); // Assuming data is in 'data/'
            // Filter for common data file types
            const dataFiles = (data.files || []).filter(f => f.match(/\.(csv|data|tsv|txt)$/i) && !f.endsWith('/'));
            setAvailableDataFiles(dataFiles.map(f => `data/${f}`)); // Store relative path from WD
            addLog(`Found data files: ${dataFiles.join(', ') || 'None'}`);
        } catch (error) {
            addLog(`Error fetching data files: ${error.message}`, 'error');
            toast.error("Could not list data files.");
        } finally {
            setDataFilesLoading(false);
        }
    }, [addLog]); // Dependency

    // Fetch data files on mount and when project changes
    useEffect(() => {
        fetchDataFiles();
    }, [fetchDataFiles, currentProject]); // Refetch when project changes

    // Update default model save path when project or task type changes
    useEffect(() => {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-'); // Basic timestamp
        setModelSavePath(`models/${currentProject || 'default'}_${taskType}_${timestamp}.pkl`);
    }, [currentProject, taskType]);

    // Determine the actual data path to use based on selection
    const effectiveDataPath = dataPathSelection === 'manual' ? manualDataPath.trim() : dataPathSelection;

    // Handle form submission
    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!effectiveDataPath || !targetColumn.trim()) {
            toast.warning("Data Path and Target Column are required for training.");
            return;
        }
        setLoading(true);
        const effectiveModelSavePath = modelSavePath.trim() || `models/default_model_${Date.now()}.pkl`; // Fallback default
        addLog(`Starting training: Data='${effectiveDataPath}', Target='${targetColumn}', Task='${taskType}', Save='${effectiveModelSavePath}'`);
        try {
            const result = await callBackendApi('/train', 'POST', {
                data_path: effectiveDataPath, // Send relative path
                target_column: targetColumn.trim(),
                model_save_path: effectiveModelSavePath, // Relative path
                task_type: taskType,
                additional_instructions: instructions,
            });
            addLog(`Training task submitted. Agent response: ${result.message}`, 'success');
            toast.success("Training task submitted successfully.");
            // Optionally clear form or update state
        } catch (err) {
            addLog(`Training failed: ${err.message}`, 'error');
            toast.error(`Training failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Card className="m-4">
            <CardHeader>
                <CardTitle>Train Model</CardTitle>
                <CardDescription>
                    Train a model on preprocessed data. Base directory: <code className="text-xs">{workingDirectory}</code>
                </CardDescription>
            </CardHeader>
            <form onSubmit={handleSubmit}>
                <CardContent className="space-y-4">
                    {/* Data Path Selection */}
                    <div>
                        <Label htmlFor="dataPathTrainSelect">Preprocessed Data Path *</Label>
                        <div className="flex gap-1 items-center">
                            <Select
                                id="dataPathTrainSelect"
                                value={dataPathSelection}
                                onChange={e => setDataPathSelection(e.target.value)}
                                disabled={dataFilesLoading}
                                required={dataPathSelection !== 'manual'} // Require selection unless manual
                                className="flex-grow"
                            >
                                <SelectItem value="" disabled>
                                    {dataFilesLoading ? "Loading data files..." :
                                     availableDataFiles.length === 0 ? "No data files found in data/" :
                                     "Select data file"}
                                </SelectItem>
                                {availableDataFiles.map(f => <SelectItem key={f} value={f}>{f}</SelectItem>)}
                                <SelectItem value="manual">Enter Path Manually...</SelectItem>
                            </Select>
                            <Button onClick={fetchDataFiles} variant="outline" size="icon" className="h-10 w-10" disabled={dataFilesLoading} aria-label="Refresh data file list">
                                <RefreshCw className={`h-4 w-4 ${dataFilesLoading ? 'animate-spin' : ''}`} />
                            </Button>
                        </div>
                        {dataPathSelection === 'manual' && (
                            <Input
                                type="text"
                                value={manualDataPath}
                                onChange={e => setManualDataPath(e.target.value)}
                                placeholder="Enter relative path to data (e.g., data/my_data.csv)"
                                required
                                className="mt-2"
                            />
                        )}
                    </div>

                    {/* Target Column */}
                    <div>
                        <Label htmlFor="targetColumnTrain">Target Column Name *</Label>
                        <Input
                            id="targetColumnTrain"
                            value={targetColumn}
                            onChange={e => setTargetColumn(e.target.value)}
                            placeholder="Enter target column name used in data"
                            required
                        />
                        <p className="text-xs text-gray-500 mt-1">Ensure this matches the target column in your dataset.</p>
                    </div>

                    {/* Task Type */}
                    <div>
                        <Label htmlFor="taskType">Task Type</Label>
                        <Select id="taskType" value={taskType} onChange={e => setTaskType(e.target.value)}>
                            <SelectItem value="classification">Classification</SelectItem>
                            <SelectItem value="regression">Regression</SelectItem>
                        </Select>
                    </div>

                    {/* Model Save Path */}
                    <div>
                        <Label htmlFor="modelPath">Save Path for Trained Model (relative)</Label>
                        <Input
                            id="modelPath"
                            value={modelSavePath}
                            onChange={(e) => setModelSavePath(e.target.value)}
                            placeholder="e.g., models/my_model.pkl"
                        />
                         <p className="text-xs text-gray-500 mt-1">Default includes project, type, and timestamp.</p>
                    </div>

                    {/* Instructions */}
                    <div>
                        <Label htmlFor="instructionsTrain">Additional Instructions (Optional)</Label>
                        <Textarea
                            id="instructionsTrain"
                            value={instructions}
                            onChange={(e) => setInstructions(e.target.value)}
                            placeholder="e.g., use SVM classifier with C=1.0, optimize for F1 score, use 5-fold cross-validation"
                            rows={3}
                        />
                    </div>
                </CardContent>
                <CardFooter>
                    <Button type="submit" disabled={loading || !effectiveDataPath || !targetColumn.trim()}>
                        {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                        Start Training
                    </Button>
                </CardFooter>
            </form>
        </Card>
    );
};


// EvaluateView Component
const EvaluateView = ({ addLog }) => {
    const { workingDirectory, currentProject } = useAppContext();
    const [availableModels, setAvailableModels] = useState([]); // Models in models/ dir
    const [selectedModel, setSelectedModel] = useState(''); // Relative path to model
    const [availableDataFiles, setAvailableDataFiles] = useState([]); // Data files in data/
    const [dataPathSelection, setDataPathSelection] = useState(''); // Dropdown state for data
    const [manualDataPath, setManualDataPath] = useState(''); // Manual input for data
    const [targetColumn, setTargetColumn] = useState('');
    const [taskType, setTaskType] = useState('classification'); // Should ideally match model
    const [evaluationPath, setEvaluationPath] = useState('results/evaluation_results.json'); // Default save path
    const [instructions, setInstructions] = useState('');
    const [loading, setLoading] = useState(false); // Main evaluation loading
    const [modelsLoading, setModelsLoading] = useState(false);
    const [dataFilesLoading, setDataFilesLoading] = useState(false);
    const [results, setResults] = useState(null); // Parsed metrics from backend
    const [rawResponse, setRawResponse] = useState(''); // Full agent response

    // Fetch models from models/ directory
    const fetchModels = useCallback(async () => {
        setModelsLoading(true);
        addLog("Fetching models from 'models/' directory...");
        setAvailableModels([]);
        try {
            const data = await callBackendApi('/files', 'GET', { path: 'models' });
            // Filter for common model file types
            const modelFiles = (data.files || []).filter(f => f.match(/\.(pkl|joblib|h5|keras|pt|pth|onnx)$/i) && !f.endsWith('/'));
            setAvailableModels(modelFiles.map(f => `models/${f}`)); // Store relative path
            addLog(`Found models: ${modelFiles.join(', ') || 'None'}`);
        } catch (error) {
            addLog(`Error fetching models: ${error.message}`, 'error');
            toast.error("Could not list models.");
        } finally {
            setModelsLoading(false);
        }
    }, [addLog]);

    // Fetch data files from data/ directory (similar to TrainView)
    const fetchDataFiles = useCallback(async () => {
        setDataFilesLoading(true);
        addLog("Fetching data files from 'data/' directory...");
        setAvailableDataFiles([]);
        try {
            const data = await callBackendApi('/files', 'GET', { path: 'data' });
            const dataFiles = (data.files || []).filter(f => f.match(/\.(csv|data|tsv|txt)$/i) && !f.endsWith('/'));
            setAvailableDataFiles(dataFiles.map(f => `data/${f}`));
            addLog(`Found data files: ${dataFiles.join(', ') || 'None'}`);
        } catch (error) {
            addLog(`Error fetching data files: ${error.message}`, 'error');
            toast.error("Could not list data files.");
        } finally {
            setDataFilesLoading(false);
        }
    }, [addLog]);

    // Fetch models and data files on mount and project change
    useEffect(() => {
        fetchModels();
        fetchDataFiles();
    }, [fetchModels, fetchDataFiles, currentProject]);

    // Determine effective data path
    const effectiveDataPath = dataPathSelection === 'manual' ? manualDataPath.trim() : dataPathSelection;

    // Handle form submission
    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!selectedModel || !effectiveDataPath || !targetColumn.trim()) {
            toast.warning("Model Path, Data Path, and Target Column are required.");
            return;
        }
        setLoading(true);
        setResults(null); // Clear previous results
        setRawResponse(''); // Clear previous raw response
        const effectiveEvalPath = evaluationPath.trim() || 'results/evaluation_results.json';
        addLog(`Starting evaluation: Model='${selectedModel}', Data='${effectiveDataPath}', Target='${targetColumn}', Task='${taskType}', Save='${effectiveEvalPath}'`);
        try {
            const result = await callBackendApi('/evaluate', 'POST', {
                model_path: selectedModel, // Relative path
                data_path: effectiveDataPath, // Relative path
                target_column: targetColumn.trim(),
                evaluation_save_path: effectiveEvalPath, // Relative path
                task_type: taskType,
                additional_instructions: instructions,
            });
            addLog(`Evaluation task submitted. Agent message received.`, 'success');
            setRawResponse(result.message || ''); // Store raw message

            if (result.results) {
                setResults(result.results); // Store parsed metrics
                addLog(`Parsed Evaluation Results: ${JSON.stringify(result.results)}`);
                toast.success("Evaluation complete. Results parsed.");
            } else {
                addLog("Evaluation complete, but could not parse structured results from agent response.", "warning");
                toast.warning("Evaluation complete, but results couldn't be parsed automatically. Check logs.");
            }
        } catch (err) {
            addLog(`Evaluation failed: ${err.message}`, 'error');
            toast.error(`Evaluation failed: ${err.message}`);
            setRawResponse(`Error: ${err.message}`); // Show error in raw output area
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col md:flex-row gap-4 m-4 items-start">
            {/* Left Panel: Form */}
            <Card className="flex-1 w-full md:w-2/3"> {/* Responsive width */}
                <CardHeader>
                    <CardTitle>Evaluate Model</CardTitle>
                    <CardDescription>
                        Evaluate model performance on a dataset. Base directory: <code className="text-xs">{workingDirectory}</code>
                    </CardDescription>
                </CardHeader>
                <form onSubmit={handleSubmit}>
                    <CardContent className="space-y-4">
                        {/* Model Selection */}
                        <div>
                            <Label htmlFor="modelSelectEval">Select Model *</Label>
                            <div className="flex gap-1 items-center">
                                <Select id="modelSelectEval" value={selectedModel} onChange={e => setSelectedModel(e.target.value)} disabled={modelsLoading || availableModels.length === 0} required className="flex-grow" >
                                    <SelectItem value="" disabled> {modelsLoading ? "Loading..." : (availableModels.length === 0 ? "No models found in models/" : "Select a model")} </SelectItem>
                                    {availableModels.map(m => <SelectItem key={m} value={m}>{m}</SelectItem>)}
                                </Select>
                                <Button onClick={fetchModels} variant="outline" size="icon" className="h-10 w-10" disabled={modelsLoading} aria-label="Refresh model list"> <RefreshCw className={`h-4 w-4 ${modelsLoading ? 'animate-spin' : ''}`} /> </Button>
                            </div>
                        </div>

                        {/* Data Path Selection */}
                        <div>
                            <Label htmlFor="dataPathEvalSelect">Evaluation Data Path *</Label>
                            <div className="flex gap-1 items-center">
                                <Select id="dataPathEvalSelect" value={dataPathSelection} onChange={e => setDataPathSelection(e.target.value)} disabled={dataFilesLoading} required={dataPathSelection !== 'manual'} className="flex-grow" >
                                    <SelectItem value="" disabled> {dataFilesLoading ? "Loading..." : (availableDataFiles.length === 0 ? "No data files found" : "Select data file")} </SelectItem>
                                    {availableDataFiles.map(f => <SelectItem key={f} value={f}>{f}</SelectItem>)}
                                    <SelectItem value="manual">Enter Path Manually...</SelectItem>
                                </Select>
                                <Button onClick={fetchDataFiles} variant="outline" size="icon" className="h-10 w-10" disabled={dataFilesLoading} aria-label="Refresh data file list"> <RefreshCw className={`h-4 w-4 ${dataFilesLoading ? 'animate-spin' : ''}`} /> </Button>
                            </div>
                            {dataPathSelection === 'manual' && (
                                <Input type="text" value={manualDataPath} onChange={e => setManualDataPath(e.target.value)} placeholder="Enter relative path to data" required className="mt-2" />
                            )}
                             <p className="text-xs text-gray-500 mt-1">Use the same dataset (or its test split) used for training/preprocessing.</p>
                        </div>

                        {/* Target Column */}
                        <div>
                            <Label htmlFor="targetColumnEval">Target Column Name *</Label>
                            <Input id="targetColumnEval" value={targetColumn} onChange={e => setTargetColumn(e.target.value)} placeholder="Target column name in the data" required/>
                        </div>

                        {/* Task Type */}
                        <div>
                            <Label htmlFor="taskTypeEval">Task Type (associated with model)</Label>
                            <Select id="taskTypeEval" value={taskType} onChange={e => setTaskType(e.target.value)}>
                                <SelectItem value="classification">Classification</SelectItem>
                                <SelectItem value="regression">Regression</SelectItem>
                            </Select>
                        </div>

                        {/* Save Path */}
                        <div>
                            <Label htmlFor="evaluationPath">Save Path for Evaluation Results (relative)</Label>
                            <Input id="evaluationPath" value={evaluationPath} onChange={(e) => setEvaluationPath(e.target.value)} placeholder="Default: results/evaluation_results.json" />
                        </div>

                        {/* Instructions */}
                        <div>
                            <Label htmlFor="instructionsEval">Additional Instructions (Optional)</Label>
                            <Textarea id="instructionsEval" value={instructions} onChange={(e) => setInstructions(e.target.value)} placeholder="e.g., evaluate using specific subset, generate confusion matrix plot instruction, report class-wise metrics" rows={3}/>
                        </div>
                    </CardContent>
                    <CardFooter>
                        <Button type="submit" disabled={loading || !selectedModel || !effectiveDataPath || !targetColumn.trim()}>
                            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                            Start Evaluation
                        </Button>
                    </CardFooter>
                </form>
            </Card>

            {/* Right Panel: Results */}
            <Card className="w-full md:w-1/3"> {/* Responsive width */}
                <CardHeader> <CardTitle>Evaluation Results</CardTitle> </CardHeader>
                <CardContent className="space-y-3">
                    {/* Parsed Metrics */}
                    {results ? (
                        <div className="space-y-2">
                            <h4 className="font-medium text-sm">Metrics:</h4>
                            <ul className="list-disc list-inside space-y-1 text-sm bg-gray-50 p-3 rounded border">
                                {Object.entries(results).map(([key, value]) => (
                                    <li key={key}>
                                        <span className="font-semibold capitalize">{key.replace(/_/g, ' ')}:</span> {/* Format key */}
                                        <span className="ml-2">{typeof value === 'number' ? value.toFixed(4) : String(value)}</span> {/* Format value */}
                                    </li>
                                ))}
                            </ul>
                            {/* Placeholder for potential chart */}
                            {/* <div className="text-center text-sm text-gray-400 border rounded p-4 mt-2">Bar Chart Placeholder</div> */}
                        </div>
                    ) : (
                        <p className="text-sm text-gray-500 italic text-center p-4">
                            {loading ? "Running evaluation..." : "Metrics will appear here after evaluation."}
                        </p>
                    )}

                    {/* Raw Agent Output */}
                    <div className="pt-2">
                        <h4 className="font-medium text-sm mb-1">Raw Agent Output:</h4>
                        <ScrollArea className="h-40 border rounded bg-gray-50 p-2 text-xs font-mono">
                            <pre className="whitespace-pre-wrap break-words">{rawResponse || (loading ? "Waiting for response..." : "No output.")}</pre>
                        </ScrollArea>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};


// PlotView Component
const PlotView = ({ addLog }) => {
    const { workingDirectory, currentProject } = useAppContext();
    const [availableSources, setAvailableSources] = useState([]); // Combined list from results/ and data/
    const [sourceSelection, setSourceSelection] = useState(''); // Dropdown state ('manual' or file path)
    const [manualSourcePath, setManualSourcePath] = useState(''); // Manual input state
    const [targetColumn, setTargetColumn] = useState(''); // Optional target for data plots
    const [instructions, setInstructions] = useState(''); // Plotting instructions
    const [loading, setLoading] = useState(false); // Main plotting action loading
    const [sourcesLoading, setSourcesLoading] = useState(false); // For fetching sources
    const [plotUrl, setPlotUrl] = useState(null); // URL/Path to the generated plot
    const [rawResponse, setRawResponse] = useState(''); // Agent's raw response

    // Fetch available sources (files in results/ and data/)
    const fetchSources = useCallback(async () => {
        setSourcesLoading(true);
        addLog("Fetching plot sources from 'results/' and 'data/'...");
        let sources = [];
        setAvailableSources([]); // Clear previous
        try {
            // Fetch from results/
            const resultsData = await callBackendApi('/files', 'GET', { path: 'results' });
            sources = sources.concat(
                (resultsData.files || [])
                    .filter(f => !f.endsWith('/')) // Only files
                    .map(f => ({ type: 'results', path: `results/${f}` })) // Add type/prefix
            );
        } catch (error) { addLog(`Error fetching results files: ${error.message}`, 'warning'); }
        try {
            // Fetch from data/
            const dataData = await callBackendApi('/files', 'GET', { path: 'data' });
            sources = sources.concat(
                (dataData.files || [])
                    .filter(f => !f.endsWith('/')) // Only files
                    .map(f => ({ type: 'data', path: `data/${f}` })) // Add type/prefix
            );
        } catch (error) { addLog(`Error fetching data files: ${error.message}`, 'warning'); }

        // Sort sources (optional, e.g., by path)
        sources.sort((a, b) => a.path.localeCompare(b.path));

        setAvailableSources(sources);
        addLog(`Found ${sources.length} potential plot sources.`);
        setSourcesLoading(false);
    }, [addLog]); // Dependency

    // Fetch sources on mount and project change
    useEffect(() => {
        fetchSources();
    }, [fetchSources, currentProject]);

    // Determine effective source path and inferred plot type
    const effectiveSourcePath = sourceSelection === 'manual' ? manualSourcePath.trim() : sourceSelection;
    // Infer plot type based on selected path prefix (simple approach) or keep 'data' as default if manual
    const inferredPlotType = effectiveSourcePath.startsWith('results/') ? 'results' : 'data';

    // Handle form submission
    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!effectiveSourcePath) {
            toast.warning("Please select or enter a data source for plotting.");
            return;
        }
        setLoading(true);
        setPlotUrl(null); // Clear previous plot
        setRawResponse(''); // Clear previous response
        addLog(`Generating plot for ${inferredPlotType} source: '${effectiveSourcePath}'`);
        try {
            const result = await callBackendApi('/plot', 'POST', {
                plot_type: inferredPlotType, // Use inferred type
                data_file_path: effectiveSourcePath, // Relative path
                target_column: targetColumn.trim() || null, // Pass target column if provided
                additional_instructions: instructions,
                // plot_save_dir is handled by backend logic
            });
            addLog(`Plot generation task submitted. Agent response: ${result.message}`, 'success');
            setRawResponse(result.message || ''); // Store raw message

            if (result.plot_path) {
                // IMPORTANT: Assuming the backend returns an ABSOLUTE path.
                // For security and browser compatibility, file:// URLs might not work directly in <img> src
                // unless the browser has specific permissions or if using Electron/Tauri.
                // A better approach for web apps is for the backend to serve the file via HTTP
                // and return a relative URL like /plots/plot_xyz.png.
                // For now, we'll try the file:// URL, but add a note about potential issues.
                const fileUrl = `file://${result.plot_path}`;
                setPlotUrl(fileUrl); // Use the absolute path provided by backend
                addLog(`Plot generated at: ${result.plot_path}. Attempting to display via ${fileUrl}`);
                toast.success("Plot generated. Displaying plot...");
                toast.info("Note: Displaying local files (file://) might be blocked by browser security.", { duration: 10000 });
            } else {
                addLog("Plot generation finished, but no plot path was found in the response.", "warning");
                toast.warning("Plot generated, but path couldn't be determined. Check logs.");
            }
        } catch (err) {
            addLog(`Plotting failed: ${err.message}`, 'error');
            toast.error(`Plotting failed: ${err.message}`);
            setRawResponse(`Error: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    // Handle image loading error
    const handlePlotError = (e) => {
        e.target.onerror = null; // Prevent infinite loop
        const failedUrl = e.target.src;
        e.target.src = `https://placehold.co/400x300/EEE/31343C?text=Error+Loading+Plot`; // Placeholder
        addLog(`Error loading plot image: ${failedUrl}. Browser might be blocking file:// access or path is incorrect.`, 'error');
        toast.error("Failed to display plot. Browser might block local file access.");
    }

    return (
        <div className="flex flex-col md:flex-row gap-4 m-4 items-start">
            {/* Left Panel: Form */}
            <Card className="flex-1 w-full md:w-3/5"> {/* Adjust width */}
                <CardHeader>
                    <CardTitle>Generate Plot</CardTitle>
                    <CardDescription>
                        Visualize data or results using plotting instructions. Base directory: <code className="text-xs">{workingDirectory}</code>
                    </CardDescription>
                </CardHeader>
                <form onSubmit={handleSubmit}>
                    <CardContent className="space-y-4">
                        {/* Data Source Selection */}
                        <div>
                            <Label htmlFor="plotSourceSelect">Data Source *</Label>
                            <div className="flex gap-1 items-center">
                                <Select
                                    id="plotSourceSelect"
                                    value={sourceSelection}
                                    onChange={e => setSourceSelection(e.target.value)}
                                    disabled={sourcesLoading}
                                    required={sourceSelection !== 'manual'}
                                    className="flex-grow"
                                >
                                    <SelectItem value="" disabled>
                                        {sourcesLoading ? "Loading sources..." :
                                         availableSources.length === 0 ? "No sources found" :
                                         "Select data or results file"}
                                    </SelectItem>
                                    {/* Group options for clarity */}
                                    <optgroup label="Results Files">
                                        {availableSources.filter(s => s.type === 'results').map(s => <SelectItem key={s.path} value={s.path}>{s.path}</SelectItem>)}
                                    </optgroup>
                                    <optgroup label="Data Files">
                                        {availableSources.filter(s => s.type === 'data').map(s => <SelectItem key={s.path} value={s.path}>{s.path}</SelectItem>)}
                                    </optgroup>
                                    <SelectItem value="manual">Enter Path Manually...</SelectItem>
                                </Select>
                                <Button onClick={fetchSources} variant="outline" size="icon" className="h-10 w-10" disabled={sourcesLoading} aria-label="Refresh source list">
                                    <RefreshCw className={`h-4 w-4 ${sourcesLoading ? 'animate-spin' : ''}`} />
                                </Button>
                            </div>
                            {sourceSelection === 'manual' && (
                                <Input
                                    type="text"
                                    value={manualSourcePath}
                                    onChange={e => setManualSourcePath(e.target.value)}
                                    placeholder="Enter relative path to source file (e.g., data/my_data.csv)"
                                    required
                                    className="mt-2"
                                />
                            )}
                        </div>

                        {/* Target Column (Only relevant for 'data' type plots) */}
                        {(sourceSelection === 'manual' || sourceSelection.startsWith('data/')) && (
                            <div>
                                <Label htmlFor="targetColumnPlot">Target Column (Optional, for data plots)</Label>
                                <Input id="targetColumnPlot" value={targetColumn} onChange={e => setTargetColumn(e.target.value)} placeholder="Enter target column name if relevant for coloring/grouping"/>
                            </div>
                        )}

                        {/* Instructions */}
                        <div>
                            <Label htmlFor="instructionsPlot">Plotting Instructions *</Label>
                            <Textarea
                                id="instructionsPlot"
                                value={instructions}
                                onChange={(e) => setInstructions(e.target.value)}
                                placeholder={
                                    inferredPlotType === 'results'
                                        ? "e.g., create a bar plot of accuracy and F1 score; plot the confusion matrix"
                                        : "e.g., create PCA plot colored by target; show correlation heatmap; plot distribution of 'feature_x'"
                                }
                                rows={4} // More space for instructions
                                required // Make instructions required for plotting
                            />
                        </div>
                    </CardContent>
                    <CardFooter>
                        <Button type="submit" disabled={loading || !effectiveSourcePath || !instructions.trim()}>
                            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                            Generate Plot
                        </Button>
                    </CardFooter>
                </form>
            </Card>

            {/* Right Panel: Plot Display */}
            <Card className="w-full md:w-2/5"> {/* Adjust width */}
                <CardHeader> <CardTitle>Generated Plot</CardTitle> </CardHeader>
                <CardContent className="space-y-3">
                    {/* Plot Image Area */}
                    <div className="min-h-[200px] border rounded flex items-center justify-center bg-gray-100"> {/* Added min-height */}
                        {plotUrl ? (
                            <img
                                src={plotUrl}
                                alt="Generated Plot"
                                className="max-w-full max-h-[400px] h-auto rounded" // Limit height
                                onError={handlePlotError} // Use error handler
                            />
                        ) : (
                            <div className="text-center text-sm text-gray-400 p-10">
                                {loading ? <Loader2 className="h-6 w-6 animate-spin mx-auto"/> : "Plot will appear here."}
                            </div>
                        )}
                    </div>
                    {/* Raw Agent Output */}
                    <div>
                        <h4 className="font-medium text-sm mb-1">Raw Agent Output:</h4>
                        <ScrollArea className="h-24 border rounded bg-gray-50 p-2 text-xs font-mono">
                            <pre className="whitespace-pre-wrap break-words">{rawResponse || (loading ? "Waiting for response..." : "No output.")}</pre>
                        </ScrollArea>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};


// CustomInstructionView Component
const CustomInstructionView = ({ addLog }) => {
    const { workingDirectory } = useAppContext();
    const [instruction, setInstruction] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(''); // Stores the agent's response/result

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!instruction.trim()) {
            toast.warning("Please enter an instruction.");
            return;
        }
        setLoading(true);
        setResult(''); // Clear previous result
        addLog(`Executing custom instruction: "${instruction}"`);
        try {
            const response = await callBackendApi('/custom', 'POST', { instruction: instruction.trim() });
            // Assuming the backend returns a simple message in the 'message' field
            addLog(`Custom instruction executed. Agent response received.`, 'success');
            setResult(response.message || "No specific result message returned."); // Display the message
            toast.success("Custom instruction executed.");
        } catch (err) {
            addLog(`Custom instruction failed: ${err.message}`, 'error');
            setResult(`Error: ${err.message}`); // Display error in the result area
            toast.error(`Execution failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Card className="m-4">
            <CardHeader>
                <CardTitle>Custom Instruction</CardTitle>
                <CardDescription>
                    Execute arbitrary Python code via the agent in the current working directory.
                    Base directory: <code className="text-xs">{workingDirectory}</code>
                </CardDescription>
            </CardHeader>
            <form onSubmit={handleSubmit}>
                <CardContent className="space-y-4">
                    <div>
                        <Label htmlFor="customInstruction">Instruction *</Label>
                        <Textarea
                            id="customInstruction"
                            value={instruction}
                            onChange={(e) => setInstruction(e.target.value)}
                            placeholder="e.g., 'Perform PCA on data/preprocessed_data.csv, save results to results/pca.csv, and plot the first two components saving to plots/pca_plot.png'. Or 'List all files in the 'models' directory'."
                            rows={6}
                            required
                        />
                         <p className="text-xs text-gray-500 mt-1">Be specific. The agent will attempt to write and run code based on this instruction.</p>
                    </div>

                    {/* Display Execution Result/Output */}
                    {(loading || result) && ( // Show area if loading or if there's a result
                        <div className="mt-4">
                             <h4 className="font-medium text-sm mb-1">Execution Output:</h4>
                            <ScrollArea className="h-40 border rounded bg-gray-50 p-2 text-sm font-mono">
                                {loading && !result && <div className="flex justify-center p-2"><Loader2 className="h-4 w-4 animate-spin"/></div>}
                                {result && <pre className="whitespace-pre-wrap break-words">{result}</pre>}
                            </ScrollArea>
                        </div>
                    )}
                </CardContent>
                <CardFooter>
                    <Button type="submit" disabled={loading || !instruction.trim()}>
                        {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                        Execute Instruction
                    </Button>
                </CardFooter>
            </form>
        </Card>
    );
};


// AutoPilotView Component
const AutoPilotView = ({ addLog }) => {
    const { workingDirectory } = useAppContext();
    const [dataPath, setDataPath] = useState(''); // Relative path to dataset
    const [targetColumn, setTargetColumn] = useState('');
    const [taskType, setTaskType] = useState('classification');
    const [iterations, setIterations] = useState(5); // Default iterations
    const [loading, setLoading] = useState(false); // Main autopilot loading
    const [plotUrl, setPlotUrl] = useState(null); // URL/Path for the final box plot
    const [rawResponse, setRawResponse] = useState(''); // Agent's raw output

    // State for the mini file browser
    const [browsePath, setBrowsePath] = useState('.');
    const [browseFiles, setBrowseFiles] = useState([]);
    const [browseLoading, setBrowseLoading] = useState(false);

    // Fetch files for the mini browser (similar to PreprocessView)
    const fetchFilesForBrowse = useCallback(async (relativePath) => {
        setBrowseLoading(true);
        try {
            const data = await callBackendApi('/files', 'GET', { path: relativePath });
             const sortedFiles = (data.files || []).sort((a, b) => { /* Sort directories first */
                 const isADir = a.endsWith('/'); const isBDir = b.endsWith('/');
                 if (isADir && !isBDir) return -1; if (!isADir && isBDir) return 1;
                 return a.localeCompare(b);
             });
            setBrowseFiles(sortedFiles);
            setBrowsePath(data.current_directory || relativePath);
        } catch (error) {
            toast.error(`Failed to browse files: ${error.message}`);
            setBrowseFiles([]);
        } finally {
            setBrowseLoading(false);
        }
    }, []);

    // Initial fetch for the browser
    useEffect(() => {
        fetchFilesForBrowse('.');
    }, [fetchFilesForBrowse]);

    // Handle selecting a file in the mini browser
    const handleFileSelect = (item) => {
        if (item.endsWith('/')) {
            const dirName = item.slice(0, -1);
            const newPath = browsePath === '.' ? dirName : `${browsePath}/${dirName}`;
            fetchFilesForBrowse(newPath);
        } else {
            const selectedFilePath = browsePath === '.' ? item : `${browsePath}/${item}`;
             if (selectedFilePath.match(/\.(csv|data|tsv|txt)$/i)) {
                setDataPath(selectedFilePath);
                addLog(`Selected dataset for Auto-Pilot: ${selectedFilePath}`);
             } else {
                 toast.warning("Please select a valid data file (e.g., .csv, .data, .tsv).");
             }
        }
    };

    // Handle going up in the mini browser
    const handleBrowseUp = () => {
        if (browsePath === '.') return;
        const parts = browsePath.split('/');
        parts.pop();
        fetchFilesForBrowse(parts.length === 0 ? '.' : parts.join('/'));
    };

    // Handle form submission
    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!dataPath || !targetColumn.trim()) {
            toast.warning("Dataset Path and Target Column are required for Auto-Pilot.");
            return;
        }
        if (iterations < 1) {
            toast.warning("Number of iterations must be at least 1.");
            return;
        }
        setLoading(true);
        setPlotUrl(null); // Clear previous plot
        setRawResponse(''); // Clear previous response
        addLog(`Starting Auto-Pilot: Data='${dataPath}', Target='${targetColumn}', Task='${taskType}', Iterations=${iterations}`);
        try {
            const result = await callBackendApi('/autopilot', 'POST', {
                data_path: dataPath,
                target_column: targetColumn.trim(),
                task_type: taskType,
                iterations: iterations,
                // plot_save_dir is handled by backend
            });
            addLog(`Auto-Pilot task submitted. Agent response: ${result.message}`, 'success');
            setRawResponse(result.message || '');

            if (result.plot_path) {
                // Similar file:// URL issue as in PlotView
                const fileUrl = `file://${result.plot_path}`;
                setPlotUrl(fileUrl);
                addLog(`Auto-Pilot plot generated at: ${result.plot_path}. Attempting display via ${fileUrl}`);
                toast.success("Auto-Pilot finished. Plot generated.");
                toast.info("Note: Displaying local files (file://) might be blocked by browser security.", { duration: 10000 });
            } else {
                addLog("Auto-Pilot finished, but no plot path was found in the response.", "warning");
                toast.warning("Auto-Pilot finished, but plot path couldn't be determined. Check logs.");
            }
        } catch (err) {
            addLog(`Auto-Pilot failed: ${err.message}`, 'error');
            toast.error(`Auto-Pilot failed: ${err.message}`);
            setRawResponse(`Error: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

     // Handle image loading error (same as PlotView)
    const handlePlotError = (e) => {
        e.target.onerror = null;
        const failedUrl = e.target.src;
        e.target.src = `https://placehold.co/400x300/EEE/31343C?text=Error+Loading+Plot`;
        addLog(`Error loading Auto-Pilot plot image: ${failedUrl}. Browser might be blocking file:// access or path is incorrect.`, 'error');
        toast.error("Failed to display plot. Browser might block local file access.");
    }

    return (
        <div className="flex flex-col md:flex-row gap-4 m-4 items-start">
            {/* Left Panel: Form */}
            <Card className="flex-1 w-full md:w-3/5"> {/* Adjust width */}
                <CardHeader>
                    <CardTitle>Auto-Pilot Workflow</CardTitle>
                    <CardDescription>
                        Run an end-to-end ML pipeline multiple times with simulated bootstrapping.
                        Base directory: <code className="text-xs">{workingDirectory}</code>
                    </CardDescription>
                </CardHeader>
                <form onSubmit={handleSubmit}>
                    <CardContent className="space-y-4">
                        {/* Dataset File Browser */}
                        <div>
                            <Label className="mb-1">Select Dataset File *</Label>
                            <div className="p-2 border rounded-md bg-gray-50 space-y-1">
                                <div className="flex gap-1 text-xs items-center mb-1">
                                    <span className="font-medium">Current:</span> <code className="bg-white px-1 rounded text-xs flex-grow truncate">{browsePath}</code>
                                    <Button onClick={handleBrowseUp} disabled={browseLoading || browsePath === '.'} variant="ghost" size="sm" className="ml-auto h-6 px-1 text-xs">.. Up</Button>
                                    <Button onClick={() => fetchFilesForBrowse(browsePath)} variant="ghost" size="icon" className="h-6 w-6" disabled={browseLoading} aria-label="Refresh browser"> <RefreshCw className={`h-3 w-3 ${browseLoading ? 'animate-spin' : ''}`} /> </Button>
                                </div>
                                <ScrollArea className="h-32 border rounded bg-white p-1">
                                    {browseLoading ? <div className="flex justify-center p-2"><Loader2 className="h-4 w-4 animate-spin"/></div> : (
                                        browseFiles.length > 0 ? (
                                            <ul> {browseFiles.map((item, i) => ( <li key={i}><button type="button" onClick={() => handleFileSelect(item)} className={`flex items-center text-xs w-full text-left p-0.5 rounded hover:bg-blue-50 group ${item.endsWith('/') ? 'text-yellow-700' : 'text-gray-800'}`}> {item.endsWith('/') ? <FolderOpen size={14} className="mr-1 shrink-0 text-yellow-600"/> : <File size={14} className="mr-1 shrink-0 text-gray-500"/>} <span className="truncate">{item}</span> </button></li> ))} </ul>
                                        ) : <p className="text-xs text-gray-400 p-2 text-center italic">Empty or error.</p>
                                    )}
                                </ScrollArea>
                                <p className="text-xs text-gray-600 mt-1">Selected: <code className="bg-white px-1 rounded font-medium">{dataPath || 'None'}</code></p>
                            </div>
                        </div>

                        {/* Target Column */}
                        <div>
                            <Label htmlFor="targetColumnAuto">Target Column Name *</Label>
                            <Input id="targetColumnAuto" value={targetColumn} onChange={e => setTargetColumn(e.target.value)} placeholder="Enter target column name" required/>
                        </div>

                        {/* Task Type */}
                        <div>
                            <Label htmlFor="taskTypeAuto">Task Type</Label>
                            <Select id="taskTypeAuto" value={taskType} onChange={e => setTaskType(e.target.value)}>
                                <SelectItem value="classification">Classification</SelectItem>
                                <SelectItem value="regression">Regression</SelectItem>
                            </Select>
                        </div>

                        {/* Number of Iterations */}
                        <div>
                            <Label htmlFor="iterationsAuto">Number of Iterations *</Label>
                            <Input id="iterationsAuto" type="number" min="1" value={iterations} onChange={e => setIterations(Math.max(1, parseInt(e.target.value, 10) || 1))} required/>
                        </div>
                    </CardContent>
                    <CardFooter>
                        <Button type="submit" disabled={loading || !dataPath || !targetColumn.trim()}>
                            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                            Run Auto-Pilot
                        </Button>
                    </CardFooter>
                </form>
            </Card>

            {/* Right Panel: Plot and Output */}
            <Card className="w-full md:w-2/5"> {/* Adjust width */}
                <CardHeader> <CardTitle>Auto-Pilot Result Plot</CardTitle> </CardHeader>
                <CardContent className="space-y-3">
                    {/* Plot Image Area */}
                    <div className="min-h-[200px] border rounded flex items-center justify-center bg-gray-100">
                        {plotUrl ? (
                            <img src={plotUrl} alt="Auto-Pilot Result Plot" className="max-w-full max-h-[400px] h-auto rounded" onError={handlePlotError} />
                        ) : (
                            <div className="text-center text-sm text-gray-400 p-10">
                                {loading ? <Loader2 className="h-6 w-6 animate-spin mx-auto"/> : "Box plot of metrics will appear here."}
                            </div>
                        )}
                    </div>
                    {/* Raw Agent Output */}
                    <div>
                        <h4 className="font-medium text-sm mb-1">Raw Agent Output:</h4>
                        <ScrollArea className="h-32 border rounded bg-gray-50 p-2 text-xs font-mono">
                            <pre className="whitespace-pre-wrap break-words">{rawResponse || (loading ? "Waiting for response..." : "No output.")}</pre>
                        </ScrollArea>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};


// SettingsView Component
const SettingsView = ({ addLog }) => {
    const {
        workingDirectory, updateWorkingDirectory, llmConfig, updateLlmConfig,
        availableOllamaModels, setAvailableOllamaModels, updateConfigStatus
    } = useAppContext();

    // Local state for inputs
    const [cwdInput, setCwdInput] = useState(workingDirectory);
    const [apiKeyInput, setApiKeyInput] = useState(''); // Keep API key input separate, don't store in main context state

    // Loading states
    const [ollamaModelsLoading, setOllamaModelsLoading] = useState(false);
    const [settingCwdLoading, setSettingCwdLoading] = useState(false);
    const [settingLlmLoading, setSettingLlmLoading] = useState(false);

    // Update local CWD input if context changes (e.g., on initial load or project selection)
    useEffect(() => {
        setCwdInput(workingDirectory);
    }, [workingDirectory]);

    // Fetch Ollama models if Ollama is selected provider
    const fetchOllamaModels = useCallback(async () => {
        setOllamaModelsLoading(true);
        addLog("Fetching available Ollama models...");
        setAvailableOllamaModels([]); // Clear previous
        try {
            // Use the endpoint currently specified in the config state
            const data = await callBackendApi('/ollama_models', 'GET', { ollama_endpoint: llmConfig.ollamaEndpoint });
            const modelNames = (data.models || []).map(m => m.name);
            setAvailableOllamaModels(modelNames);
            addLog(`Found ${modelNames.length} Ollama models.`);
            toast.success("Ollama models loaded successfully.");
            // Auto-select the first model if current selection is invalid or empty
            if (modelNames.length > 0 && !modelNames.includes(llmConfig.ollamaModel)) {
                 updateLlmConfig({ ollamaModel: modelNames[0] });
            } else if (modelNames.length === 0) {
                 updateLlmConfig({ ollamaModel: '' }); // Clear selection if no models found
            }
        } catch (err) {
            addLog(`Error fetching Ollama models: ${err.message}`, 'error');
            toast.error(`Failed Ollama fetch: ${err.message}`);
        } finally {
            setOllamaModelsLoading(false);
        }
    }, [addLog, setAvailableOllamaModels, llmConfig.ollamaEndpoint, llmConfig.ollamaModel, updateLlmConfig]); // Added dependencies

    // Fetch Ollama models initially if Ollama is the default provider
    useEffect(() => {
        if (llmConfig.provider === 'ollama') {
            fetchOllamaModels();
        }
        // No cleanup needed for fetch
    }, [llmConfig.provider, fetchOllamaModels]); // Rerun if provider changes or fetch function updates

    // Handle setting the working directory
    const handleSetWorkingDirectory = async () => {
        const pathInput = cwdInput.trim();
        if (!pathInput) {
            toast.warning("Please enter a directory path.");
            return;
        }
        setSettingCwdLoading(true);
        addLog(`Attempting to set working directory to: ${pathInput}`);
        updateConfigStatus('cwd', 'pending');
        try {
            const result = await callBackendApi('/set_working_directory', 'POST', { path: pathInput });
            updateWorkingDirectory(result.path); // Update context with confirmed path from backend
            setCwdInput(result.path); // Update local input to match confirmed path
            addLog(`Working directory set to: ${result.path}`, 'success');
            toast.success(result.message);
            updateConfigStatus('cwd', 'ok');
        } catch (err) {
            addLog(`Failed to set working directory: ${err.message}`, 'error');
            toast.error(`CWD Error: ${err.message}`);
            updateConfigStatus('cwd', 'error');
            // Optionally revert cwdInput to workingDirectory from context on error
            // setCwdInput(workingDirectory);
        } finally {
            setSettingCwdLoading(false);
        }
    };

    // Handle saving LLM configuration
    const handleSaveLlmConfig = async () => {
        setSettingLlmLoading(true);
        addLog(`Attempting to set LLM config: Provider=${llmConfig.provider}`);
        updateConfigStatus('llm', 'pending');

        // Prepare config payload, including API key only if OpenAI is selected
        const configToSend = {
            provider: llmConfig.provider,
            api_key: llmConfig.provider === 'openai' ? apiKeyInput : null, // Send key only for OpenAI
            api_model: llmConfig.apiModel,
            ollama_model: llmConfig.ollamaModel,
            ollama_endpoint: llmConfig.ollamaEndpoint,
        };

        try {
            const result = await callBackendApi('/set_llm_config', 'POST', configToSend);
            addLog(`LLM configuration saved: ${result.message}`, 'success');
            toast.success(result.message);
            updateConfigStatus('llm', 'ok');
            // Clear the API key input field after successful save for security
            if (llmConfig.provider === 'openai') {
                setApiKeyInput('');
            }
        } catch (err) {
            addLog(`Failed to save LLM config: ${err.message}`, 'error');
            toast.error(`LLM Config Error: ${err.message}`);
            updateConfigStatus('llm', 'error');
        } finally {
            setSettingLlmLoading(false);
        }
    };

    return (
        <div className="p-4 space-y-6">
            {/* Working Directory Card */}
            <Card>
                <CardHeader>
                    <CardTitle>Working Directory</CardTitle>
                    <CardDescription>Set the root directory for all project operations and file browsing.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                    <Label htmlFor="cwdInput">Directory Path</Label>
                    <div className="flex gap-2">
                        <Input
                            id="cwdInput"
                            value={cwdInput}
                            onChange={(e) => setCwdInput(e.target.value)}
                            placeholder="/path/to/your/projects/folder"
                            className="flex-grow"
                        />
                        <Button onClick={handleSetWorkingDirectory} disabled={settingCwdLoading}>
                            {settingCwdLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                            Set Directory
                        </Button>
                    </div>
                    <p className="text-xs text-gray-500">Current: <code className="bg-gray-100 px-1 rounded">{workingDirectory}</code></p>
                </CardContent>
            </Card>

            {/* LLM Configuration Card */}
            <Card>
                <CardHeader>
                    <CardTitle>LLM Configuration</CardTitle>
                    <CardDescription>Select and configure the Language Model provider for the agent.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    {/* Provider Selection */}
                    <div>
                        <Label>LLM Provider</Label>
                        <div className="flex gap-4 mt-1 text-sm">
                            <label className="flex items-center cursor-pointer p-2 rounded hover:bg-gray-100">
                                <input
                                    type="radio"
                                    name="llmProvider"
                                    value="openai"
                                    checked={llmConfig.provider === 'openai'}
                                    onChange={() => updateLlmConfig({ provider: 'openai' })}
                                    className="mr-2"
                                />
                                OpenAI API
                            </label>
                            <label className="flex items-center cursor-pointer p-2 rounded hover:bg-gray-100">
                                <input
                                    type="radio"
                                    name="llmProvider"
                                    value="ollama"
                                    checked={llmConfig.provider === 'ollama'}
                                    onChange={() => updateLlmConfig({ provider: 'ollama' })}
                                    className="mr-2"
                                />
                                Ollama (Local)
                            </label>
                        </div>
                    </div>

                    {/* OpenAI Specific Settings */}
                    {llmConfig.provider === 'openai' && (
                        <div className="p-4 border rounded-md space-y-4 bg-blue-50/30 border-blue-200">
                            <h4 className="font-medium text-blue-800">OpenAI Settings</h4>
                            {/* Security Warning */}
                            <div className="p-3 border border-yellow-300 bg-yellow-50 rounded-md">
                                <p className="text-xs text-yellow-800 flex items-start gap-1.5">
                                    <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0"/>
                                    <span><strong>Security Warning:</strong> API keys provide access to your account. Handle them securely. Ensure your backend environment reads keys securely (e.g., from environment variables). Avoid exposing keys in frontend code.</span>
                                </p>
                            </div>
                            {/* API Key Input */}
                            <div>
                                <Label htmlFor="apiKey" className="flex items-center gap-1"><KeyRound size={14}/> API Key</Label>
                                <Input
                                    id="apiKey"
                                    type="password"
                                    value={apiKeyInput}
                                    onChange={e => setApiKeyInput(e.target.value)}
                                    placeholder="Enter your OpenAI API Key (sk-...)"
                                />
                                 <p className="text-xs text-gray-500 mt-1">Key is sent to backend on save, not stored in browser.</p>
                            </div>
                            {/* Model Selection */}
                            <div>
                                <Label htmlFor="apiModel">Model</Label>
                                <Select
                                    id="apiModel"
                                    value={llmConfig.apiModel}
                                    onChange={e => updateLlmConfig({ apiModel: e.target.value })}
                                >
                                    {/* Add more models as needed */}
                                    <SelectItem value="gpt-4o">gpt-4o (Recommended)</SelectItem>
                                    <SelectItem value="gpt-4-turbo">gpt-4-turbo</SelectItem>
                                    <SelectItem value="gpt-3.5-turbo">gpt-3.5-turbo</SelectItem>
                                </Select>
                            </div>
                        </div>
                    )}

                    {/* Ollama Specific Settings */}
                    {llmConfig.provider === 'ollama' && (
                        <div className="p-4 border rounded-md space-y-4 bg-green-50/30 border-green-200">
                            <h4 className="font-medium text-green-800">Ollama Settings</h4>
                            <p className="text-xs text-gray-600">Requires Ollama to be running locally or accessible via the specified endpoint.</p>
                            {/* Ollama Endpoint */}
                            <div>
                                <Label htmlFor="ollamaEndpoint" className="flex items-center gap-1"><Server size={14}/> Ollama Server Endpoint</Label>
                                <Input
                                    id="ollamaEndpoint"
                                    value={llmConfig.ollamaEndpoint}
                                    onChange={e => updateLlmConfig({ ollamaEndpoint: e.target.value })}
                                    placeholder="e.g., http://localhost:11434"
                                />
                            </div>
                            {/* Ollama Model Selection */}
                            <div>
                                <Label htmlFor="ollamaModel">Model</Label>
                                <div className="flex gap-2 items-center">
                                    <Select
                                        id="ollamaModel"
                                        value={llmConfig.ollamaModel}
                                        onChange={e => updateLlmConfig({ ollamaModel: e.target.value })}
                                        disabled={ollamaModelsLoading || availableOllamaModels.length === 0}
                                        className="flex-grow"
                                    >
                                        <SelectItem value="" disabled>
                                            {ollamaModelsLoading ? "Loading models..." :
                                             availableOllamaModels.length === 0 ? "No models found/error" :
                                             "Select a model"}
                                        </SelectItem>
                                        {availableOllamaModels.map(modelName => (
                                            <SelectItem key={modelName} value={modelName}>{modelName}</SelectItem>
                                        ))}
                                    </Select>
                                    <Button onClick={fetchOllamaModels} variant="outline" size="icon" className="h-10 w-10" disabled={ollamaModelsLoading} aria-label="Refresh Ollama models">
                                        <RefreshCw className={`h-4 w-4 ${ollamaModelsLoading ? 'animate-spin' : ''}`} />
                                    </Button>
                                </div>
                                 <p className="text-xs text-gray-500 mt-1">Click refresh after changing endpoint or if Ollama models update.</p>
                            </div>
                        </div>
                    )}
                </CardContent>
                <CardFooter>
                    <Button onClick={handleSaveLlmConfig} disabled={settingLlmLoading}>
                        {settingLlmLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                        Save LLM Configuration
                    </Button>
                </CardFooter>
            </Card>
        </div>
    );
};


// Log Panel Component
const LogPanel = ({ logs, clearLogs }) => {
    const scrollAreaRef = useRef(null);

    // Auto-scroll to bottom when logs update
    useEffect(() => {
        if (scrollAreaRef.current) {
            scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
        }
    }, [logs]); // Dependency: logs array

    // Helper to get log message color based on type
    const getLogColor = (type) => {
        switch (type) {
            case 'error': return 'text-red-600';
            case 'warning': return 'text-yellow-600';
            case 'success': return 'text-green-600';
            default: return 'text-gray-700'; // info or default
        }
    };

    return (
        <div className="h-48 border-t border-gray-200 bg-gray-50 flex flex-col"> {/* Fixed height */}
            {/* Log Panel Header */}
            <div className="flex justify-between items-center p-2 border-b bg-white sticky top-0"> {/* Sticky header */}
                <h3 className="text-sm font-medium text-gray-600">Activity Log</h3>
                <Button variant="ghost" size="sm" onClick={clearLogs} className="text-xs h-7 px-2"> {/* Smaller button */}
                    <X className="h-3 w-3 mr-1"/> Clear Logs
                </Button>
            </div>

            {/* Log Messages Area */}
            <ScrollArea className="flex-grow p-2" ref={scrollAreaRef}>
                {logs.length === 0 ? (
                    <p className="text-sm text-gray-400 italic p-2">No activity yet.</p>
                ) : (
                    logs.map((log, index) => (
                        <p key={index} className={`text-xs font-mono ${getLogColor(log.type)} leading-relaxed`}> {/* Improved line spacing */}
                            <span className="text-gray-400 mr-2 select-none">{log.timestamp}</span> {/* Non-selectable timestamp */}
                            {log.message}
                        </p>
                    ))
                )}
            </ScrollArea>
        </div>
    );
};


// Main App Component (Wrapper for Context and Initial Setup)
function App() {
    const [currentView, setCurrentView] = useState('welcome'); // Start with welcome view
    const [logs, setLogs] = useState([]);

    // Function to add a log entry
    const addLog = useCallback((message, type = 'info') => {
        const timestamp = new Date().toLocaleTimeString();
        setLogs(prevLogs => [...prevLogs, { timestamp, message, type }]);
        // Also log to console for debugging
        switch(type) {
            case 'error': console.error(`[${timestamp}] ${message}`); break;
            case 'warning': console.warn(`[${timestamp}] ${message}`); break;
            default: console.log(`[${timestamp}] ${message}`);
        }
    }, []); // No dependencies needed

    // Function to clear logs
    const clearLogs = useCallback(() => {
        setLogs([]);
        addLog("Logs cleared.");
    }, [addLog]); // Dependency: addLog

    // Function to check backend status on initial load
    const checkBackendStatus = useCallback(async (updateConfigStatus, updateWorkingDirectory) => {
        addLog("Checking backend status...");
        try {
            const data = await callBackendApi('/status'); // Use GET by default
            addLog(`Backend status: ${data.message}`, 'success');
            // Update config status based on backend response
            if (data.config_status) {
                updateConfigStatus('llm', data.config_status.llm || 'pending');
                updateConfigStatus('cwd', data.config_status.cwd || 'pending');
            } else {
                 updateConfigStatus('llm', 'pending'); // Default if not provided
                 updateConfigStatus('cwd', 'pending');
            }
            // Update working directory based on backend response
            if (data.working_directory) {
                updateWorkingDirectory(data.working_directory);
            }
             toast.success("Connected to backend.");
        } catch (error) {
            addLog(`Failed to connect to backend: ${error.message}`, 'error');
            updateConfigStatus('llm', 'error'); // Mark as error if status check fails
            updateConfigStatus('cwd', 'error');
            toast.error(`Backend Connection Error: ${error.message}`);
        }
    }, [addLog]); // Dependency: addLog

    return (
        // Provide App context to children
        <AppProvider>
            {/* Pass state and functions down to the inner component */}
            <AppInner
                currentView={currentView}
                setCurrentView={setCurrentView}
                logs={logs}
                addLog={addLog}
                clearLogs={clearLogs}
                checkBackendStatus={checkBackendStatus}
            />
        </AppProvider>
    );
}

// Inner component to consume context and handle initial effect
function AppInner({ currentView, setCurrentView, logs, addLog, clearLogs, checkBackendStatus }) {
    const { updateConfigStatus, updateWorkingDirectory } = useAppContext(); // Get context updaters

    // Check backend status on initial mount
    useEffect(() => {
        addLog("ML Copilot GUI Initialized.");
        checkBackendStatus(updateConfigStatus, updateWorkingDirectory);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Run only once on mount

    return (
        // Main layout container
        <div className="flex h-screen bg-white font-sans antialiased text-gray-900">
            {/* Ensure Inter font is loaded (usually via index.html or index.css) */}
            <style>{`body { font-family: 'Inter', sans-serif; }`}</style>

            {/* Sidebar */}
            <Sidebar currentView={currentView} setView={setCurrentView} addLog={addLog} />

            {/* Main Content Area */}
            <main className="flex-1 flex flex-col overflow-hidden">
                {/* Scrollable content area */}
                <div className="flex-1 overflow-y-auto bg-gray-100">
                    <MainContent currentView={currentView} addLog={addLog} clearLogs={clearLogs} />
                </div>
                {/* Log Panel at the bottom */}
                <LogPanel logs={logs} clearLogs={clearLogs} />
            </main>

            {/* Toaster for notifications */}
            <Toaster position="top-right" richColors />
        </div>
    );
}

export default App;
