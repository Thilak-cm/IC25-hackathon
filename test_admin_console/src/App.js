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
function CategoryTable({ categoryName, categoryData, onDelete }) {
  if (!categoryData) return null;

  if (categoryName === 'Allowed' || categoryName === 'Not Allowed') {
    const rows = [];
    Object.entries(categoryData).forEach(([lot, ruleObj]) => {
      const endDay = ruleObj['End Day'] || '';
      const endTime = ruleObj['End Time'] || '';
      Object.entries(ruleObj).forEach(([key, value]) => {
        if (key === 'End Day' || key === 'End Time') return;
        rows.push({
          lot,
          timeFrame: key,
          permits: Array.isArray(value) ? value.join(', ') : '',
          endDay,
          endTime,
        });
      });
    });
    return (
      <table className="table">
        <thead>
          <tr>
            <th className="th">Lot</th>
            <th className="th">Time Frame</th>
            <th className="th">Permits</th>
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

  if (categoryName === 'Closed') {
    const rows = Object.entries(categoryData).map(([lot, obj]) => ({
      lot,
      endDay: obj['End Day'] || '',
      endTime: obj['End Time'] || '',
    }));
    return (
      <table className="table">
        <thead>
          <tr>
            <th className="th">Lot</th>
            <th className="th">End Day</th>
            <th className="th">End Time</th>
            <th className="th deleteColumn deleteHeader"></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="td">{r.lot}</td>
              <td className="td">{r.endDay}</td>
              <td className="td">{r.endTime}</td>
              <td className="td deleteColumn">
                <button
                  className="deleteButton"
                  onClick={() =>
                    onDelete({ category: categoryName, lot: r.lot })
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

  if (categoryName === 'New Permits') {
    const rows = Object.entries(categoryData).map(([name, obj]) => ({
      name,
      perms: (obj['Perms'] || []).join(', '),
      endDay: obj['End Day'] || '',
      endTime: obj['End Time'] || '',
    }));
    return (
      <table className="table">
        <thead>
          <tr>
            <th className="th">Name</th>
            <th className="th">Permits</th>
            <th className="th">End Day</th>
            <th className="th">End Time</th>
            <th className="th deleteColumn deleteHeader"></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="td">{r.name}</td>
              <td className="td">{r.perms}</td>
              <td className="td">{r.endDay}</td>
              <td className="td">{r.endTime}</td>
              <td className="td deleteColumn">
                <button
                  className="deleteButton"
                  onClick={() =>
                    onDelete({
                      category: categoryName,
                      new_permit_name: r.name,
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

  return null;
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
  const [scheduledRules, setScheduledRules] = useState(null);
  const [alertLogs, setAlertLogs] = useState([]);
  
  // Toggles for displaying rules tables
  const [showRules, setShowRules] = useState(false);
  const [showScheduled, setShowScheduled] = useState(false);
  
  // Which category's rules to display in the rules table
  const [rulesCategoryToShow, setRulesCategoryToShow] = useState('Allowed');
  
  // Notifications (vertical stack on right)
  const [notifications, setNotifications] = useState([]);
  const prevAlertIdsRef = useRef([]);
  
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
        if (showRules) fetchRules();
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
    if (!inEffectFrom || !inEffectTo) {
      addNotification('error', 'Please fill "In Effect From" and "In Effect To".');
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
      const res = await fetch('http://localhost:1000/update_rule', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        addNotification('success', data.status || 'OK');
        if (showRules) fetchRules();
        if (showScheduled) fetchScheduledRules();
      } else {
        addNotification('error', JSON.stringify(data));
      }
    } catch (err) {
      addNotification('error', err.message);
    }
  };
  
  /*****************************************************************
   * Fetch functions
   *****************************************************************/
  const fetchRules = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:1000/get_restrictions');
      const data = await res.json();
      setRules(data);
    } catch (err) {
      addNotification('error', `Error fetching rules: ${err.message}`);
    }
  }, [addNotification]);
  
  const fetchScheduledRules = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:1000/pending_updates');
      const data = await res.json();
      setScheduledRules(data);
    } catch (err) {
      addNotification('error', `Error fetching scheduled rules: ${err.message}`);
    }
  }, [addNotification]);
  
  const fetchAlerts = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:1000/get_alerts');
      const data = await res.json();
      if (!data.length) {
        prevAlertIdsRef.current = [];
      } else {
        const newIds = data.map((a) => a.id);
        // If new alerts are found
        if (
          !prevAlertIdsRef.current.length ||
          (newIds.length > prevAlertIdsRef.current.length &&
            newIds[0] !== prevAlertIdsRef.current[0])
        ) {
          addNotification('error', `New Alert: ${data[0].alert_message}`);
          if (showRules) fetchRules();
          if (showScheduled) fetchScheduledRules();
        }
        prevAlertIdsRef.current = newIds;
      }
      setAlertLogs(data);
    } catch (err) {
      addNotification('error', `Error fetching alerts: ${err.message}`);
    }
  }, [addNotification, showRules, showScheduled, fetchRules, fetchScheduledRules]);
  
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
   * Toggle display for rules tables
   *****************************************************************/
  const toggleRules = async () => {
    if (!showRules) {
      await fetchRules();
    }
    setShowRules((prev) => !prev);
  };
  
  const toggleScheduled = async () => {
    if (!showScheduled) {
      await fetchScheduledRules();
    }
    setShowScheduled((prev) => !prev);
  };
  
  /*****************************************************************
   * On mount, fetch alerts periodically
   *****************************************************************/
  useEffect(() => {
    fetchAlerts();
    const t = setInterval(fetchAlerts, 5000);
    return () => clearInterval(t);
  }, [fetchAlerts]);
  
  /*****************************************************************
   * Render
   *****************************************************************/
  return (
    <div className="layoutContainer">
      <div className="leftPanel">
        <div className="leftPanelOverlay">
          <div className="overlayBox">
            <img src="assets/logo.png" alt="Logo" className="overlayLogo" />
            <h2 className="overlayHeading">Parking Management Rule Updater</h2>
          </div>
        </div>
        <img
          src="assets/image1.webp"
          alt="Descriptive visual"
          className="leftPanelImage"
        />
      </div>
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
  
            <button type="submit" className="button">
              Submit Rule
            </button>
          </form>
  
          <div className="buttonContainer">
            <button onClick={toggleRules} className="button">
              {showRules ? 'Hide Current Rules' : 'Fetch Current Rules'}
            </button>
            <button onClick={toggleScheduled} className="button">
              {showScheduled ? 'Hide Scheduled Rules' : 'Show Scheduled Rules'}
            </button>
          </div>
  
          {showRules && rules && (
            <div className="dataContainer">
              <h2 className="header1">Current Rules</h2>
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
              />
            </div>
          )}
  
          {showScheduled && scheduledRules && (
            <div className="dataContainer">
              <h2 className="header1">Scheduled Rules (Pending Updates)</h2>
              <table className="table">
                <thead>
                  <tr>
                    <th className="th">Category</th>
                    <th className="th">Lot/Name</th>
                    <th className="th">Time Slot</th>
                    <th className="th">Perms</th>
                    <th className="th">In Effect From</th>
                    <th className="th">End Day</th>
                    <th className="th">End Time</th>
                    <th className="th">Delete</th>
                  </tr>
                </thead>
                <tbody>
                  {scheduledRules.map((item, i) => {
                    const ts = Array.isArray(item.time_slot)
                      ? item.time_slot.join('|')
                      : item.time_slot;
                    const permsStr = Array.isArray(item.perms)
                      ? item.perms.join(', ')
                      : '';
                    return (
                      <tr key={i}>
                        <td className="td">{item.category}</td>
                        <td className="td">{item.lot}</td>
                        <td className="td">{ts}</td>
                        <td className="td">{permsStr}</td>
                        <td className="td">{item.in_effect_from}</td>
                        <td className="td">{item.end_day}</td>
                        <td className="td">{item.end_time}</td>
                        <td className="td">
                          <button
                            className="deleteButton"
                            onClick={() =>
                              deleteRule({
                                category: item.category,
                                lot: item.lot,
                                time_frame: ts,
                              })
                            }
                          >
                            &#10005;
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
  
          <div className="dataContainer">
            <div className="alertLogHeader">
              <h2 className="header1">Alert Logs</h2>
              <button className="button" onClick={clearAllAlerts}>
                Clear All Alerts
              </button>
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
                {alertLogs.map((a) => (
                  <tr key={a.id} className="alertRow">
                    <td className="td">{a.timestamp}</td>
                    <td className="td">
                      {a.alert_message} <span className="alertId">(ID: {a.id})</span>
                    </td>
                    <td className="td">{a.details}</td>
                    <td className="td">
                      <button className="deleteButton" onClick={() => deleteAlert(a.id)}>
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