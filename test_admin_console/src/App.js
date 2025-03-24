import React, { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';

/*****************************************************************
 * ChipMultiSelect component 
 *****************************************************************/
function ChipMultiSelect({ options, selected, setSelected, placeholder }) {
  const [inputValue, setInputValue] = useState('');
  const opts = options || [];

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && inputValue.trim()) {
      e.preventDefault();
      addChip(inputValue.trim());
    }
  };

  const addChip = (val) => {
    if (!selected.includes(val) && opts.includes(val)) {
      setSelected((prev) => [...prev, val]);
    }
    setInputValue('');
  };

  const removeChip = (val) => {
    setSelected((prev) => prev.filter((x) => x !== val));
  };

  const filteredOptions = opts.filter((o) =>
    o.toLowerCase().includes(inputValue.toLowerCase())
  );

  return (
    <div className="chipContainer">
      <div className="chipSelectedContainer">
        {selected.map((item) => (
          <div key={item} className="chipSelected">
            {item}
            <span className="chipRemove" onClick={() => removeChip(item)}>
              x
            </span>
          </div>
        ))}
      </div>
      <input
        className="chipInput"
        placeholder={placeholder}
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      {inputValue && (
        <div className="chipDropdown">
          {filteredOptions.slice(0, 10).map((opt) => (
            <div
              key={opt}
              className="chipDropdownItem"
              onClick={() => addChip(opt)}
            >
              {opt}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/*****************************************************************
 * Notification component (vertical stack on right)
 *****************************************************************/
function Notification({ notification }) {
  return (
    <div className={`notification ${notification.type}`}>
      {notification.type === 'error' ? '❗' : '✅'} {notification.message}
    </div>
  );
}

/*****************************************************************
 * CategoryTable component for showing rules with Delete button
 *****************************************************************/
function CategoryTable({ categoryName, categoryData, onDelete, viewMode }) {
  console.log('CategoryTable received:', {
    categoryName,
    categoryData,
    viewMode
  });

  if (!categoryData) return null;

  const rows = [];
  Object.entries(categoryData).forEach(([lot, ruleObj]) => {
    // Special handling for Closed category
    if (categoryName === 'Closed') {
      if (viewMode === 'active' && ruleObj.active) {
        rows.push({
          lot,
          timeFrame: 'Full Day', // Closed lots are closed for the full day
          permits: 'None',
          endDay: ruleObj.active['End Day'],
          endTime: ruleObj.active['End Time'],
          status: 'active'
        });
      }
      if (viewMode === 'pending' && ruleObj.pending) {
        rows.push({
          lot,
          timeFrame: 'Full Day',
          permits: 'None',
          endDay: ruleObj.pending.end_day,
          endTime: ruleObj.pending.end_time,
          status: 'pending'
        });
      }
      return;
    }

    // Existing code for other categories
    if (viewMode === 'active' && ruleObj.active) {
      const endDay = ruleObj.active['End Day'] || '';
      const endTime = ruleObj.active['End Time'] || '';
      Object.entries(ruleObj.active).forEach(([key, value]) => {
        if (key === 'End Day' || key === 'End Time') return;
        rows.push({
          lot,
          timeFrame: key,
          permits: Array.isArray(value) ? value.join(', ') : '',
          endDay,
          endTime,
          status: 'active'
        });
      });
    }
    if (viewMode === 'pending' && ruleObj.pending) {
      rows.push({
        lot,
        timeFrame: Array.isArray(ruleObj.pending.time_slot) 
          ? ruleObj.pending.time_slot.join('|') 
          : ruleObj.pending.time_slot,
        permits: Array.isArray(ruleObj.pending.perms) 
          ? ruleObj.pending.perms.join(', ') 
          : '',
        endDay: ruleObj.pending.end_day,
        endTime: ruleObj.pending.end_time,
        startDay: ruleObj.pending.Start_Day,
        startTime: ruleObj.pending.Start_Time,
        status: 'pending'
      });
    }
  });

  if (rows.length === 0) {
    return (
      <div className="emptyStateMessage">
        {`No ${viewMode === 'active' ? 'active' : 'scheduled'} rules in this category`}
      </div>
    );
  }

  // Adjust columns based on view mode
  return (
    <table className="table">
      <thead>
        <tr>
          <th className="th">Lot</th>
          <th className="th">Time Frame</th>
          <th className="th">Permits</th>
          {viewMode === 'pending' && (
            <>
              <th className="th">Start Day</th>
              <th className="th">Start Time</th>
            </>
          )}
          <th className="th">End Day</th>
          <th className="th">End Time</th>
          <th className="th deleteColumn deleteHeader"></th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            <td className="td">{r.lot}</td>
            <td className="td">{r.timeFrame}</td>
            <td className="td">{r.permits}</td>
            {viewMode === 'pending' && (
              <>
                <td className="td">{r.startDay}</td>
                <td className="td">{r.startTime}</td>
              </>
            )}
            <td className="td">{r.endDay}</td>
            <td className="td">{r.endTime}</td>
            <td className="td deleteColumn">
              <button
                className="deleteButton"
                onClick={() =>
                  onDelete({
                    category: categoryName,
                    lot: r.lot,
                    time_frame: r.timeFrame,
                    status: r.status
                  })
                }
              >
                &#10005;
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/*****************************************************************
 * Main App Component with Two-Column Layout
 *****************************************************************/
function App() {
  // Dynamic dropdown data fetched from backend
  const [lotNames, setLotNames] = useState([]);
  const [permitNames, setPermitNames] = useState([]);
  
  useEffect(() => {
    fetch('http://localhost:1000/get_lot_perms')
      .then((res) => res.json())
      .then((data) => {
        setLotNames(data.LOT_NAMES || []);
        setPermitNames(data.PERM_NAMES || []);
      })
      .catch((err) => console.error('Error fetching lot/permit names:', err));
  }, []);
  
  // Category selection for submission
  const [category, setCategory] = useState('Allowed');
  
  // For Allowed/Not Allowed
  const [lots, setLots] = useState([]);
  const [permits, setPermits] = useState([]);
  const [enforceStart, setEnforceStart] = useState('');
  const [enforceEnd, setEnforceEnd] = useState('');
  
  // For New Permits
  const [newPermitName, setNewPermitName] = useState('');
  
  // In Effect From and In Effect To
  const [inEffectFrom, setInEffectFrom] = useState('');
  const [inEffectTo, setInEffectTo] = useState('');
  
  // Backend data for rules and scheduled updates
  const [rules, setRules] = useState(null);
  const [alertLogs, setAlertLogs] = useState([]);
  
  // Which category's rules to display in the rules table
  const [rulesCategoryToShow, setRulesCategoryToShow] = useState('Allowed');
  
  // Notifications (vertical stack on right)
  const [notifications, setNotifications] = useState([]);
  const prevAlertIdsRef = useRef([]);
  
  // 1. Change state to track which view we want
  const [viewMode, setViewMode] = useState('none'); // 'none', 'active', or 'pending'
  
  // Add state for tracking divider position
  const [isDragging, setIsDragging] = useState(false);
  const [leftPanelWidth, setLeftPanelWidth] = useState(700); // Initial width
  const dividerRef = useRef(null);
  
  // Add these state variables at the top of your App component
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  
  // Add state for filter type and custom limit
  const [filterType, setFilterType] = useState('none'); // 'none', 'latest', 'date'
  const [latestCount, setLatestCount] = useState(5); // Default to 5
  
  // First add this state at the top of your App component
  const [errorMessage, setErrorMessage] = useState('');
  
  /*****************************************************************
   * Utility functions
   *****************************************************************/
  const addNotification = useCallback((type, message) => {
    const id = Date.now();
    setNotifications((prev) => [...prev, { id, type, message }]);
    setTimeout(() => {
      setNotifications((prev) => prev.filter((n) => n.id !== id));
    }, 3000);
  }, []);
  
  // Convert datetime-local to "YYYY-MM-DD HH:MM:SS"
  const formatDateTime = (dtStr) => {
    if (!dtStr) return '';
    let out = dtStr.replace('T', ' ');
    if (out.length === 16) out += ':00';
    return out;
  };
  
  // Convert "HH:MM" to "HH:MM:SS"
  const formatTime = (str) => {
    if (!str) return '';
    return str.length === 5 ? str + ':00' : str;
  };
  
  /*****************************************************************
   * Delete Rule handler – called from table rows
   *****************************************************************/
  const deleteRule = async (payload) => {
    try {
      const res = await fetch('http://localhost:1000/delete_rule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        addNotification('success', 'Rule deleted');
        if (viewMode !== 'none') fetchRules();
      } else {
        addNotification('error', JSON.stringify(data));
      }
    } catch (err) {
      addNotification('error', err.message);
    }
  };
  
  /*****************************************************************
   * Submit handler
   *****************************************************************/
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Clear any existing error message
    setErrorMessage('');
    
    // Validate lot selection
    if (lots.length === 0) {
      setErrorMessage('Error: Please select at least one lot');
      return;
    }

    // Validate dates are provided
    if (!inEffectFrom || !inEffectTo) {
      setErrorMessage('Error: Please fill in both date fields');
      return;
    }

    // Validate date order
    const fromDate = new Date(inEffectFrom);
    const toDate = new Date(inEffectTo);
    if (fromDate >= toDate) {
      setErrorMessage('Error: "In Effect From" must be before "In Effect To"');
      return;
    }

    let end_day = '';
    let end_time = '';
    const parts = inEffectTo.split('T');
    if (parts.length === 2) {
      end_day = parts[0];
      end_time = formatTime(parts[1]);
    } else {
      addNotification('error', 'Invalid "In Effect To" format.');
      return;
    }
    const payload = {
      category,
      in_effect_from: formatDateTime(inEffectFrom),
      end_day,
      end_time,
    };
    try {
      if (category === 'Allowed' || category === 'Not Allowed') {
        if (!enforceStart || !enforceEnd) {
          addNotification('error', 'Please fill Enforcement Hours (start/end).');
          return;
        }
        payload.time_slot = `${formatTime(enforceStart)}|${formatTime(enforceEnd)}`;
        payload.lots = lots;
        payload.perms = permits;
      } else if (category === 'Closed') {
        payload.lots = lots;
      } else if (category === 'New Permits') {
        if (!newPermitName) {
          addNotification('error', 'Please fill "Name" for new permit.');
          return;
        }
        payload.new_permit_name = newPermitName;
        payload.perms = permits;
      } else {
        addNotification('error', `Unknown category: ${category}`);
        return;
      }
    } catch (err) {
      addNotification('error', `Error building payload: ${err.message}`);
      return;
    }
    try {
      // When the user clicks "Submit Rule"
      const res = await fetch('http://localhost:1000/update_rule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        addNotification('success', data.status === 'pending update scheduled' ? 'Scheduled Rule Updated' : 'Rule Updated');
        if (viewMode !== 'none') fetchRules();
      } else {
        setErrorMessage(`Error: ${JSON.stringify(data)}`);
      }
    } catch (err) {
      setErrorMessage(`Error: ${err.message}`);
    }
  };
  
  /*****************************************************************
   * Fetch functions
   *****************************************************************/
  // When the user clicks "Fetch Current Rules"
  const fetchRules = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:1000/get_new_rules');
      const data = await res.json();
      setRules(data);
    } catch (err) {
      addNotification('error', `Error fetching rules: ${err.message}`);
    }
  }, [addNotification]);

  // When the user clicks "Fetch Alerts"
  const fetchAlerts = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:1000/get_alerts');
      const data = await res.json();
      setAlertLogs(data);
      
      // Only show notifications for actual new alerts with valid messages
      const currentAlertIds = data.map(alert => alert.id);
      const newAlertIds = currentAlertIds.filter(id => !prevAlertIdsRef.current.includes(id));
      
      // Only show one notification for the newest alert if there are any new ones
      const newestAlert = data.find(alert => alert.id === Math.max(...newAlertIds));
      if (newestAlert) {
        // Only show notification if we have a valid message
        const message = `New ${newestAlert.category} rule for lot ${newestAlert.lot}`;
        addNotification('info', message);
      }
      
      prevAlertIdsRef.current = currentAlertIds;
    } catch (err) {
      addNotification('error', `Error fetching alerts: ${err.message}`);
    }
  }, [addNotification]);

  // When the user clicks "Delete Alert"
  const deleteAlert = useCallback(
    async (alertId) => {
      try {
        const res = await fetch(`http://localhost:1000/delete_alert/${alertId}`, {
          method: 'DELETE',
        });
        const d = await res.json();
        if (d.status === 'deleted') {
          addNotification('success', `Alert ${alertId} removed`);
          fetchAlerts();
        } else {
          addNotification('error', `Delete failed: ${JSON.stringify(d)}`);
        }
      } catch (err) {
        addNotification('error', `Error deleting alert: ${err.message}`);
      }
    },
    [addNotification, fetchAlerts]
  );
  
  /*****************************************************************
   * Clear All Alerts
   *****************************************************************/
  const clearAllAlerts = async () => {
    try {
      const res = await fetch('http://localhost:1000/clear_alerts', {
        method: 'DELETE',
      });
      const d = await res.json();
      if (d.status === 'cleared') {
        addNotification('success', 'All alerts cleared');
        fetchAlerts();
      } else {
        addNotification('error', `Clear failed: ${JSON.stringify(d)}`);
      }
    } catch (err) {
      addNotification('error', `Error clearing alerts: ${err.message}`);
    }
  };
  
  /*****************************************************************
   * Update the toggle functions
   *****************************************************************/
  const toggleActiveRules = () => {
    if (viewMode === 'active') {
      setViewMode('none');
    } else {
      setViewMode('active');
      if (!rules) fetchRules();
    }
  };

  const togglePendingRules = () => {
    if (viewMode === 'pending') {
      setViewMode('none');
    } else {
      setViewMode('pending');
      if (!rules) fetchRules();
    }
  };
  
  // Add event handlers for divider dragging
  const handleMouseDown = (e) => {
    e.preventDefault(); // Prevent text selection
    setIsDragging(true);
    document.body.classList.add('dragging'); // Add dragging class to body
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const handleMouseMove = useCallback((e) => {
    if (!isDragging) return;
    
    const newWidth = Math.max(300, Math.min(e.clientX, window.innerWidth - 400));
    setLeftPanelWidth(newWidth);
  }, [isDragging]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseMove]);
  
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);
  
  /*****************************************************************
   * Add a function to filter logs
   *****************************************************************/
  const getFilteredLogs = () => {
    let filtered = [...alertLogs];
    
    if (filterType === 'latest') {
      filtered = filtered
        .sort((a, b) => b.id - a.id)
        .slice(0, latestCount);
    } else if (filterType === 'date') {
      if (startDate) {
        filtered = filtered.filter(log => new Date(log.timestamp) >= new Date(startDate));
      }
      if (endDate) {
        filtered = filtered.filter(log => new Date(log.timestamp) <= new Date(endDate));
      }
    }
    
    return filtered;
  };
  
  /*****************************************************************
   * Render
   *****************************************************************/
  return (
    <div className="layoutContainer">
      <div 
        className="leftPanel" 
        style={{ width: `${leftPanelWidth}px` }}
      >
        <div className="leftPanelOverlay">
          <div className="overlayBox">
            <img src="assets/logo.png" alt="Logo" className="overlayLogo" />
          </div>
        </div>
        <img
          src="assets/image1.webp"
          alt="Descriptive visual"
          className="leftPanelImage"
        />
      </div>
      <div 
        className="divider"
        onMouseDown={handleMouseDown}
        ref={dividerRef}
      />
      <div className="rightPanel">
        <div className="container">
          <h1 className="header">Parking Management Rule Updater</h1>
  
          <form className="form" onSubmit={handleSubmit}>
            <label className="label">Category:</label>
            <select
              className="input"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              <option value="Allowed">Allowed</option>
              <option value="Not Allowed">Not Allowed</option>
              <option value="Closed">Closed</option>
              <option value="New Permits">New Permits</option>
            </select>
  
            {(category === 'Allowed' || category === 'Not Allowed') && (
              <>
                <label className="label">Lots (Chips)</label>
                <ChipMultiSelect
                  options={lotNames}
                  selected={lots}
                  setSelected={setLots}
                  placeholder="Search or type lot"
                />
                <label className="label">Permits (Chips)</label>
                <ChipMultiSelect
                  options={permitNames}
                  selected={permits}
                  setSelected={setPermits}
                  placeholder="Search or type permit"
                />
                <label className="label">Enforcement Hours (Start)</label>
                <input
                  type="time"
                  className="input"
                  value={enforceStart}
                  onChange={(e) => setEnforceStart(e.target.value)}
                />
                <label className="label">Enforcement Hours (End)</label>
                <input
                  type="time"
                  className="input"
                  value={enforceEnd}
                  onChange={(e) => setEnforceEnd(e.target.value)}
                />
              </>
            )}
  
            {category === 'Closed' && (
              <>
                <label className="label">Lots (Chips)</label>
                <ChipMultiSelect
                  options={lotNames}
                  selected={lots}
                  setSelected={setLots}
                  placeholder="Search or type lot"
                />
              </>
            )}
  
            {category === 'New Permits' && (
              <>
                <label className="label">Name:</label>
                <input
                  type="text"
                  className="input"
                  value={newPermitName}
                  onChange={(e) => setNewPermitName(e.target.value)}
                />
                <label className="label">Permits (Chips)</label>
                <ChipMultiSelect
                  options={permitNames}
                  selected={permits}
                  setSelected={setPermitNames}
                  placeholder="Search or type permit"
                />
              </>
            )}
  
            <label className="label">In Effect From:</label>
            <input
              type="datetime-local"
              className="input"
              value={inEffectFrom}
              onChange={(e) => setInEffectFrom(e.target.value)}
            />
  
            <label className="label">In Effect To:</label>
            <input
              type="datetime-local"
              className="input"
              value={inEffectTo}
              onChange={(e) => setInEffectTo(e.target.value)}
            />
  
            <div style={{ marginBottom: '1rem' }}>
              <button type="submit" className="button">
                Submit Rule
              </button>
              {errorMessage && (
                <div style={{ 
                  color: '#dc3545',
                  marginTop: '8px',
                  fontSize: '14px',
                  fontWeight: '500',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  <span style={{ fontSize: '18px' }}>⚠️</span>
                  {errorMessage.replace('Error: ', '')}
                </div>
              )}
            </div>
          </form>
  
          <div className="buttonContainer">
            <button onClick={toggleActiveRules} className="button">
              {viewMode === 'active' ? 'Hide Current Rules' : 'Show Current Rules'}
            </button>
            <button onClick={togglePendingRules} className="button">
              {viewMode === 'pending' ? 'Hide Scheduled Rules' : 'Show Scheduled Rules'}
            </button>
          </div>
  
          {viewMode !== 'none' && rules && (
            <div className="dataContainer">
              <h2 className="header1">
                {viewMode === 'active' ? 'Current Rules' : 'Scheduled Rules'}
              </h2>
              <div className="selectCategoryRow">
                <label>Select Category:</label>
                <select
                  className="categorySelect"
                  value={rulesCategoryToShow}
                  onChange={(e) => setRulesCategoryToShow(e.target.value)}
                >
                  <option value="Allowed">Allowed</option>
                  <option value="Not Allowed">Not Allowed</option>
                  <option value="Closed">Closed</option>
                  <option value="New Permits">New Permits</option>
                </select>
              </div>
              <CategoryTable
                categoryName={rulesCategoryToShow}
                categoryData={rules[rulesCategoryToShow]}
                onDelete={deleteRule}
                viewMode={viewMode}
              />
            </div>
          )}
  
          <div className="dataContainer">
            <div className="alertLogHeader">
              <h2>Alert Logs</h2>
              <div className="filterControls">
                <div className="filterLabel">Filter by:</div>
                <select 
                  className="filterSelect"
                  value={filterType}
                  onChange={(e) => {
                    setFilterType(e.target.value);
                    setStartDate('');
                    setEndDate('');
                  }}
                >
                  <option value="none">None</option>
                  <option value="latest">Latest Entries</option>
                  <option value="date">Date Range</option>
                </select>
                
                {filterType === 'latest' && (
                  <select 
                    className="limitSelect"
                    value={latestCount}
                    onChange={(e) => setLatestCount(Number(e.target.value))}
                  >
                    <option value={5}>Latest 5</option>
                    <option value={10}>Latest 10</option>
                  </select>
                )}
                
                {filterType === 'date' && (
                  <div className="dateFilterGroup">
                    <input
                      type="datetime-local"
                      className="dateFilter"
                      placeholder="Start Date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                    />
                    <input
                      type="datetime-local"
                      className="dateFilter"
                      placeholder="End Date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                    />
                  </div>
                )}
                
                <button className="clearAlertsButton" onClick={clearAllAlerts}>
                  Clear All Alerts
                </button>
              </div>
            </div>
            <table className="table">
              <thead>
                <tr>
                  <th className="th">Timestamp</th>
                  <th className="th">Message</th>
                  <th className="th">Details</th>
                  <th className="th">Delete</th>
                </tr>
              </thead>
              <tbody>
                {getFilteredLogs().map((log) => (
                  <tr key={log.id} className="alertRow">
                    <td className="td">{log.timestamp}</td>
                    <td className="td">
                      {log.alert_message} <span className="alertId">(ID: {log.id})</span>
                    </td>
                    <td className="td">{log.details}</td>
                    <td className="td">
                      <button className="deleteButton" onClick={() => deleteAlert(log.id)}>
                        &#10005;
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
  
          <div className="notificationContainer">
            {notifications.map((n) => (
              <Notification key={n.id} notification={n} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
  
export default App;